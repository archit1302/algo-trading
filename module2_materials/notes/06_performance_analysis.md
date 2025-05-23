# Module 2.6: Performance Analysis and Risk Assessment

## Introduction

Performance analysis is crucial for evaluating trading strategies and investment portfolios. This module covers comprehensive performance metrics, risk assessment techniques, and methods for analyzing and improving trading strategies.

## Learning Objectives

By the end of this lesson, you will be able to:
- Calculate comprehensive performance metrics
- Analyze risk-adjusted returns
- Understand and calculate various risk measures
- Perform portfolio attribution analysis
- Create performance reports and visualizations
- Identify areas for strategy improvement

## 1. Return Analysis

### 1.1 Types of Returns

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Generate sample portfolio performance data
np.random.seed(42)
dates = pd.date_range('2023-01-01', periods=252, freq='B')
strategy_returns = np.random.normal(0.0012, 0.018, len(dates))
benchmark_returns = np.random.normal(0.0008, 0.015, len(dates))

performance_data = pd.DataFrame({
    'Date': dates,
    'Strategy_Returns': strategy_returns,
    'Benchmark_Returns': benchmark_returns
}, index=dates)

# Calculate cumulative returns
performance_data['Strategy_Cumulative'] = (1 + performance_data['Strategy_Returns']).cumprod() - 1
performance_data['Benchmark_Cumulative'] = (1 + performance_data['Benchmark_Returns']).cumprod() - 1

# Calculate excess returns
performance_data['Excess_Returns'] = (
    performance_data['Strategy_Returns'] - performance_data['Benchmark_Returns']
)
performance_data['Excess_Cumulative'] = (1 + performance_data['Excess_Returns']).cumprod() - 1

print("Performance Data Overview:")
print(performance_data.head())
print(f"\nStrategy Total Return: {performance_data['Strategy_Cumulative'].iloc[-1]:.4f}")
print(f"Benchmark Total Return: {performance_data['Benchmark_Cumulative'].iloc[-1]:.4f}")
print(f"Excess Return: {performance_data['Excess_Cumulative'].iloc[-1]:.4f}")
```

### 1.2 Annualized Returns and Compounding

```python
def calculate_annualized_return(returns, periods_per_year=252):
    """Calculate annualized return from periodic returns"""
    total_return = (1 + returns).prod() - 1
    n_periods = len(returns)
    annualized = (1 + total_return) ** (periods_per_year / n_periods) - 1
    return annualized

def calculate_geometric_mean(returns):
    """Calculate geometric mean return"""
    return (1 + returns).prod() ** (1 / len(returns)) - 1

# Calculate various return measures
strategy_annual = calculate_annualized_return(performance_data['Strategy_Returns'])
benchmark_annual = calculate_annualized_return(performance_data['Benchmark_Returns'])
strategy_geometric = calculate_geometric_mean(performance_data['Strategy_Returns'])

print("Return Analysis:")
print(f"Strategy Annualized Return: {strategy_annual:.4f}")
print(f"Benchmark Annualized Return: {benchmark_annual:.4f}")
print(f"Strategy Geometric Mean (Daily): {strategy_geometric:.6f}")
print(f"Strategy Arithmetic Mean (Daily): {performance_data['Strategy_Returns'].mean():.6f}")
```

## 2. Risk Metrics

### 2.1 Volatility and Standard Deviation

```python
def calculate_volatility_metrics(returns, periods_per_year=252):
    """Calculate various volatility measures"""
    daily_vol = returns.std()
    annualized_vol = daily_vol * np.sqrt(periods_per_year)
    
    # Rolling volatility
    rolling_vol = returns.rolling(window=30).std() * np.sqrt(periods_per_year)
    
    # Downside volatility (only negative returns)
    downside_returns = returns[returns < 0]
    downside_vol = downside_returns.std() * np.sqrt(periods_per_year)
    
    return {
        'Daily_Volatility': daily_vol,
        'Annualized_Volatility': annualized_vol,
        'Rolling_Volatility': rolling_vol,
        'Downside_Volatility': downside_vol
    }

# Calculate volatility metrics
strategy_vol = calculate_volatility_metrics(performance_data['Strategy_Returns'])
benchmark_vol = calculate_volatility_metrics(performance_data['Benchmark_Returns'])

print("Volatility Analysis:")
print(f"Strategy Annualized Volatility: {strategy_vol['Annualized_Volatility']:.4f}")
print(f"Benchmark Annualized Volatility: {benchmark_vol['Annualized_Volatility']:.4f}")
print(f"Strategy Downside Volatility: {strategy_vol['Downside_Volatility']:.4f}")

# Plot rolling volatility
plt.figure(figsize=(12, 6))
plt.plot(performance_data.index, strategy_vol['Rolling_Volatility'], 
         label='Strategy', linewidth=2)
plt.plot(performance_data.index, benchmark_vol['Rolling_Volatility'], 
         label='Benchmark', linewidth=2)
plt.title('Rolling 30-Day Annualized Volatility')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### 2.2 Drawdown Analysis

```python
def calculate_drawdown(returns):
    """Calculate drawdown metrics"""
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    
    # Maximum drawdown
    max_drawdown = drawdown.min()
    
    # Find drawdown periods
    is_drawdown = drawdown < 0
    drawdown_periods = []
    
    start_idx = None
    for i, is_dd in enumerate(is_drawdown):
        if is_dd and start_idx is None:
            start_idx = i
        elif not is_dd and start_idx is not None:
            drawdown_periods.append((start_idx, i-1))
            start_idx = None
    
    # Calculate recovery times
    recovery_times = []
    for start, end in drawdown_periods:
        recovery_times.append(end - start + 1)
    
    avg_recovery_time = np.mean(recovery_times) if recovery_times else 0
    
    return {
        'Drawdown_Series': drawdown,
        'Max_Drawdown': max_drawdown,
        'Avg_Recovery_Time': avg_recovery_time,
        'Drawdown_Periods': len(drawdown_periods)
    }

# Calculate drawdown metrics
strategy_dd = calculate_drawdown(performance_data['Strategy_Returns'])
benchmark_dd = calculate_drawdown(performance_data['Benchmark_Returns'])

print("Drawdown Analysis:")
print(f"Strategy Max Drawdown: {strategy_dd['Max_Drawdown']:.4f}")
print(f"Benchmark Max Drawdown: {benchmark_dd['Max_Drawdown']:.4f}")
print(f"Strategy Avg Recovery Time: {strategy_dd['Avg_Recovery_Time']:.1f} days")
print(f"Strategy Drawdown Periods: {strategy_dd['Drawdown_Periods']}")

# Plot cumulative returns and drawdowns
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Cumulative returns
ax1.plot(performance_data.index, performance_data['Strategy_Cumulative'], 
         label='Strategy', linewidth=2)
ax1.plot(performance_data.index, performance_data['Benchmark_Cumulative'], 
         label='Benchmark', linewidth=2)
ax1.set_title('Cumulative Returns')
ax1.set_ylabel('Cumulative Return')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Drawdowns
ax2.fill_between(performance_data.index, strategy_dd['Drawdown_Series'], 
                 0, alpha=0.3, color='red', label='Strategy')
ax2.fill_between(performance_data.index, benchmark_dd['Drawdown_Series'], 
                 0, alpha=0.3, color='blue', label='Benchmark')
ax2.set_title('Drawdown Analysis')
ax2.set_xlabel('Date')
ax2.set_ylabel('Drawdown')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## 3. Risk-Adjusted Performance Metrics

### 3.1 Sharpe Ratio and Variations

```python
def calculate_sharpe_ratio(returns, risk_free_rate=0.02, periods_per_year=252):
    """Calculate Sharpe ratio"""
    excess_returns = returns - risk_free_rate / periods_per_year
    return (excess_returns.mean() * periods_per_year) / (returns.std() * np.sqrt(periods_per_year))

def calculate_sortino_ratio(returns, risk_free_rate=0.02, periods_per_year=252):
    """Calculate Sortino ratio (uses downside deviation)"""
    excess_returns = returns - risk_free_rate / periods_per_year
    downside_returns = excess_returns[excess_returns < 0]
    downside_std = downside_returns.std() * np.sqrt(periods_per_year)
    return (excess_returns.mean() * periods_per_year) / downside_std

def calculate_calmar_ratio(returns, periods_per_year=252):
    """Calculate Calmar ratio (annual return / max drawdown)"""
    annual_return = calculate_annualized_return(returns, periods_per_year)
    max_dd = calculate_drawdown(returns)['Max_Drawdown']
    return annual_return / abs(max_dd) if max_dd != 0 else np.inf

# Calculate risk-adjusted metrics
risk_free_rate = 0.02  # 2% annual risk-free rate

strategy_sharpe = calculate_sharpe_ratio(performance_data['Strategy_Returns'], risk_free_rate)
benchmark_sharpe = calculate_sharpe_ratio(performance_data['Benchmark_Returns'], risk_free_rate)

strategy_sortino = calculate_sortino_ratio(performance_data['Strategy_Returns'], risk_free_rate)
benchmark_sortino = calculate_sortino_ratio(performance_data['Benchmark_Returns'], risk_free_rate)

strategy_calmar = calculate_calmar_ratio(performance_data['Strategy_Returns'])
benchmark_calmar = calculate_calmar_ratio(performance_data['Benchmark_Returns'])

print("Risk-Adjusted Performance Metrics:")
print(f"Strategy Sharpe Ratio: {strategy_sharpe:.4f}")
print(f"Benchmark Sharpe Ratio: {benchmark_sharpe:.4f}")
print(f"Strategy Sortino Ratio: {strategy_sortino:.4f}")
print(f"Benchmark Sortino Ratio: {benchmark_sortino:.4f}")
print(f"Strategy Calmar Ratio: {strategy_calmar:.4f}")
print(f"Benchmark Calmar Ratio: {benchmark_calmar:.4f}")
```

### 3.2 Information Ratio and Tracking Error

```python
def calculate_information_ratio(portfolio_returns, benchmark_returns):
    """Calculate Information Ratio"""
    excess_returns = portfolio_returns - benchmark_returns
    tracking_error = excess_returns.std() * np.sqrt(252)
    excess_annual = excess_returns.mean() * 252
    return excess_annual / tracking_error if tracking_error != 0 else np.inf

def calculate_tracking_error(portfolio_returns, benchmark_returns, periods_per_year=252):
    """Calculate tracking error"""
    excess_returns = portfolio_returns - benchmark_returns
    return excess_returns.std() * np.sqrt(periods_per_year)

def calculate_beta(portfolio_returns, benchmark_returns):
    """Calculate beta (systematic risk measure)"""
    covariance = np.cov(portfolio_returns, benchmark_returns)[0][1]
    benchmark_variance = np.var(benchmark_returns)
    return covariance / benchmark_variance

# Calculate relative performance metrics
info_ratio = calculate_information_ratio(
    performance_data['Strategy_Returns'], 
    performance_data['Benchmark_Returns']
)
tracking_error = calculate_tracking_error(
    performance_data['Strategy_Returns'], 
    performance_data['Benchmark_Returns']
)
beta = calculate_beta(
    performance_data['Strategy_Returns'], 
    performance_data['Benchmark_Returns']
)

print("Relative Performance Metrics:")
print(f"Information Ratio: {info_ratio:.4f}")
print(f"Tracking Error: {tracking_error:.4f}")
print(f"Beta: {beta:.4f}")
```

## 4. Value at Risk (VaR) and Expected Shortfall

### 4.1 Historical VaR

```python
def calculate_var_historical(returns, confidence_level=0.05):
    """Calculate Historical Value at Risk"""
    return np.percentile(returns, confidence_level * 100)

def calculate_expected_shortfall(returns, confidence_level=0.05):
    """Calculate Expected Shortfall (Conditional VaR)"""
    var = calculate_var_historical(returns, confidence_level)
    return returns[returns <= var].mean()

# Calculate VaR metrics
var_95 = calculate_var_historical(performance_data['Strategy_Returns'], 0.05)
var_99 = calculate_var_historical(performance_data['Strategy_Returns'], 0.01)
es_95 = calculate_expected_shortfall(performance_data['Strategy_Returns'], 0.05)
es_99 = calculate_expected_shortfall(performance_data['Strategy_Returns'], 0.01)

print("Value at Risk Analysis:")
print(f"1-Day VaR (95%): {var_95:.4f}")
print(f"1-Day VaR (99%): {var_99:.4f}")
print(f"Expected Shortfall (95%): {es_95:.4f}")
print(f"Expected Shortfall (99%): {es_99:.4f}")

# VaR visualization
plt.figure(figsize=(12, 6))
plt.hist(performance_data['Strategy_Returns'], bins=50, alpha=0.7, density=True)
plt.axvline(var_95, color='red', linestyle='--', linewidth=2, label=f'VaR 95%: {var_95:.4f}')
plt.axvline(var_99, color='darkred', linestyle='--', linewidth=2, label=f'VaR 99%: {var_99:.4f}')
plt.axvline(es_95, color='orange', linestyle=':', linewidth=2, label=f'ES 95%: {es_95:.4f}')
plt.title('Return Distribution with VaR and Expected Shortfall')
plt.xlabel('Daily Returns')
plt.ylabel('Density')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### 4.2 Parametric VaR

```python
def calculate_var_parametric(returns, confidence_level=0.05):
    """Calculate Parametric VaR assuming normal distribution"""
    mean = returns.mean()
    std = returns.std()
    z_score = stats.norm.ppf(confidence_level)
    return mean + z_score * std

def calculate_var_cornish_fisher(returns, confidence_level=0.05):
    """Calculate Modified VaR using Cornish-Fisher expansion"""
    mean = returns.mean()
    std = returns.std()
    skewness = stats.skew(returns)
    kurtosis = stats.kurtosis(returns)
    
    z = stats.norm.ppf(confidence_level)
    z_cf = (z + 
            (z**2 - 1) * skewness / 6 + 
            (z**3 - 3*z) * kurtosis / 24 - 
            (2*z**3 - 5*z) * skewness**2 / 36)
    
    return mean + z_cf * std

# Compare VaR methods
var_hist_95 = calculate_var_historical(performance_data['Strategy_Returns'], 0.05)
var_param_95 = calculate_var_parametric(performance_data['Strategy_Returns'], 0.05)
var_cf_95 = calculate_var_cornish_fisher(performance_data['Strategy_Returns'], 0.05)

print("VaR Method Comparison (95% confidence):")
print(f"Historical VaR: {var_hist_95:.4f}")
print(f"Parametric VaR: {var_param_95:.4f}")
print(f"Cornish-Fisher VaR: {var_cf_95:.4f}")

# Distribution statistics
print(f"\nReturn Distribution Statistics:")
print(f"Skewness: {stats.skew(performance_data['Strategy_Returns']):.4f}")
print(f"Kurtosis: {stats.kurtosis(performance_data['Strategy_Returns']):.4f}")
print(f"Jarque-Bera Test p-value: {stats.jarque_bera(performance_data['Strategy_Returns'])[1]:.4f}")
```

## 5. Performance Attribution Analysis

### 5.1 Return Attribution

```python
def performance_attribution(portfolio_returns, factor_returns):
    """Simple performance attribution analysis"""
    # Assume we have sector/factor exposure data
    sectors = ['Technology', 'Healthcare', 'Finance', 'Energy', 'Consumer']
    
    # Generate sample factor returns and weights
    np.random.seed(42)
    factor_data = pd.DataFrame({
        sector: np.random.normal(0.0008, 0.012, len(portfolio_returns))
        for sector in sectors
    }, index=portfolio_returns.index)
    
    # Sample portfolio weights (should sum to 1)
    weights = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
    
    # Calculate attribution
    attribution = pd.DataFrame(index=portfolio_returns.index)
    for i, sector in enumerate(sectors):
        attribution[f'{sector}_Return'] = factor_data[sector] * weights[i]
    
    attribution['Total_Attribution'] = attribution.sum(axis=1)
    attribution['Selection_Effect'] = portfolio_returns - attribution['Total_Attribution']
    
    return attribution, factor_data, weights

# Perform attribution analysis
attribution, factors, weights = performance_attribution(
    performance_data['Strategy_Returns'], 
    performance_data['Benchmark_Returns']
)

print("Performance Attribution Summary:")
for i, sector in enumerate(['Technology', 'Healthcare', 'Finance', 'Energy', 'Consumer']):
    sector_contrib = (attribution[f'{sector}_Return'].sum() * 252)
    print(f"{sector} Contribution: {sector_contrib:.4f} (Weight: {weights[i]:.1%})")

selection_effect = attribution['Selection_Effect'].sum() * 252
print(f"Stock Selection Effect: {selection_effect:.4f}")
```

### 5.2 Risk Attribution

```python
def risk_attribution(returns, factor_returns, weights):
    """Calculate risk attribution to different factors"""
    # Calculate covariance matrix
    factor_cov = factor_returns.cov() * 252  # Annualize
    
    # Portfolio variance from factors
    portfolio_var_factors = np.dot(weights, np.dot(factor_cov, weights))
    
    # Individual factor contributions to risk
    marginal_contrib = np.dot(factor_cov, weights)
    contrib_to_risk = weights * marginal_contrib
    
    # Specific risk (idiosyncratic)
    total_var = returns.var() * 252
    specific_var = total_var - portfolio_var_factors
    
    risk_attribution = pd.DataFrame({
        'Factor': ['Technology', 'Healthcare', 'Finance', 'Energy', 'Consumer', 'Specific'],
        'Weight': list(weights) + [1.0],
        'Contribution_to_Variance': list(contrib_to_risk) + [specific_var],
        'Contribution_to_Risk': list(np.sqrt(contrib_to_risk)) + [np.sqrt(specific_var)]
    })
    
    risk_attribution['Percent_of_Risk'] = (
        risk_attribution['Contribution_to_Variance'] / total_var
    )
    
    return risk_attribution

# Calculate risk attribution
risk_attrib = risk_attribution(
    performance_data['Strategy_Returns'], 
    factors, 
    weights
)

print("Risk Attribution Analysis:")
print(risk_attrib.round(4))
```

## 6. Rolling Performance Analysis

### 6.1 Rolling Metrics

```python
def calculate_rolling_metrics(returns, window=63):  # ~3 months
    """Calculate rolling performance metrics"""
    rolling_metrics = pd.DataFrame(index=returns.index)
    
    # Rolling returns
    rolling_metrics['Rolling_Return'] = (
        (1 + returns).rolling(window=window).apply(lambda x: x.prod() - 1)
    )
    
    # Rolling Sharpe ratio
    rolling_metrics['Rolling_Sharpe'] = (
        returns.rolling(window=window).mean() / returns.rolling(window=window).std() * np.sqrt(252)
    )
    
    # Rolling volatility
    rolling_metrics['Rolling_Volatility'] = (
        returns.rolling(window=window).std() * np.sqrt(252)
    )
    
    # Rolling max drawdown
    def rolling_max_dd(x):
        cum_ret = (1 + x).cumprod()
        running_max = cum_ret.expanding().max()
        dd = (cum_ret - running_max) / running_max
        return dd.min()
    
    rolling_metrics['Rolling_Max_DD'] = (
        returns.rolling(window=window).apply(rolling_max_dd)
    )
    
    return rolling_metrics

# Calculate rolling metrics
rolling_perf = calculate_rolling_metrics(performance_data['Strategy_Returns'])

print("Rolling Performance Metrics (Last 10 observations):")
print(rolling_perf.tail(10).round(4))

# Plot rolling metrics
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

axes[0, 0].plot(rolling_perf.index, rolling_perf['Rolling_Return'])
axes[0, 0].set_title('Rolling 3-Month Returns')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(rolling_perf.index, rolling_perf['Rolling_Sharpe'])
axes[0, 1].set_title('Rolling 3-Month Sharpe Ratio')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(rolling_perf.index, rolling_perf['Rolling_Volatility'])
axes[1, 0].set_title('Rolling 3-Month Volatility')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(rolling_perf.index, rolling_perf['Rolling_Max_DD'])
axes[1, 1].set_title('Rolling 3-Month Max Drawdown')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## 7. Comprehensive Performance Report

### 7.1 Performance Summary

```python
def create_performance_report(strategy_returns, benchmark_returns, risk_free_rate=0.02):
    """Create comprehensive performance report"""
    
    report = {}
    
    # Basic metrics
    report['Performance Metrics'] = {
        'Total Return': (1 + strategy_returns).prod() - 1,
        'Annualized Return': calculate_annualized_return(strategy_returns),
        'Volatility': strategy_returns.std() * np.sqrt(252),
        'Sharpe Ratio': calculate_sharpe_ratio(strategy_returns, risk_free_rate),
        'Sortino Ratio': calculate_sortino_ratio(strategy_returns, risk_free_rate),
        'Calmar Ratio': calculate_calmar_ratio(strategy_returns)
    }
    
    # Risk metrics
    dd_metrics = calculate_drawdown(strategy_returns)
    report['Risk Metrics'] = {
        'Max Drawdown': dd_metrics['Max_Drawdown'],
        'VaR (95%)': calculate_var_historical(strategy_returns, 0.05),
        'Expected Shortfall (95%)': calculate_expected_shortfall(strategy_returns, 0.05),
        'Skewness': stats.skew(strategy_returns),
        'Kurtosis': stats.kurtosis(strategy_returns)
    }
    
    # Relative metrics
    report['Relative Metrics'] = {
        'Beta': calculate_beta(strategy_returns, benchmark_returns),
        'Information Ratio': calculate_information_ratio(strategy_returns, benchmark_returns),
        'Tracking Error': calculate_tracking_error(strategy_returns, benchmark_returns),
        'Active Return': (strategy_returns - benchmark_returns).mean() * 252
    }
    
    # Trading metrics (if applicable)
    win_rate = len(strategy_returns[strategy_returns > 0]) / len(strategy_returns[strategy_returns != 0])
    report['Trading Metrics'] = {
        'Win Rate': win_rate,
        'Average Win': strategy_returns[strategy_returns > 0].mean(),
        'Average Loss': strategy_returns[strategy_returns < 0].mean(),
        'Best Day': strategy_returns.max(),
        'Worst Day': strategy_returns.min()
    }
    
    return report

# Generate comprehensive report
performance_report = create_performance_report(
    performance_data['Strategy_Returns'],
    performance_data['Benchmark_Returns']
)

print("COMPREHENSIVE PERFORMANCE REPORT")
print("=" * 50)

for category, metrics in performance_report.items():
    print(f"\n{category}:")
    print("-" * 30)
    for metric, value in metrics.items():
        if isinstance(value, float):
            if abs(value) < 0.01:
                print(f"{metric}: {value:.6f}")
            else:
                print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")
```

### 7.2 Performance Visualization Dashboard

```python
def create_performance_dashboard(strategy_returns, benchmark_returns):
    """Create comprehensive performance visualization dashboard"""
    
    fig = plt.figure(figsize=(16, 12))
    
    # Cumulative returns
    ax1 = plt.subplot(3, 3, 1)
    strategy_cum = (1 + strategy_returns).cumprod() - 1
    benchmark_cum = (1 + benchmark_returns).cumprod() - 1
    plt.plot(strategy_returns.index, strategy_cum, label='Strategy', linewidth=2)
    plt.plot(benchmark_returns.index, benchmark_cum, label='Benchmark', linewidth=2)
    plt.title('Cumulative Returns')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Rolling Sharpe ratio
    ax2 = plt.subplot(3, 3, 2)
    rolling_sharpe = calculate_rolling_metrics(strategy_returns)['Rolling_Sharpe']
    plt.plot(rolling_sharpe.index, rolling_sharpe)
    plt.title('Rolling 3-Month Sharpe Ratio')
    plt.grid(True, alpha=0.3)
    
    # Drawdown
    ax3 = plt.subplot(3, 3, 3)
    dd = calculate_drawdown(strategy_returns)['Drawdown_Series']
    plt.fill_between(strategy_returns.index, dd, 0, alpha=0.3, color='red')
    plt.title('Drawdown')
    plt.grid(True, alpha=0.3)
    
    # Return distribution
    ax4 = plt.subplot(3, 3, 4)
    plt.hist(strategy_returns, bins=30, alpha=0.7, density=True)
    plt.axvline(strategy_returns.mean(), color='red', linestyle='--', label='Mean')
    plt.title('Return Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Monthly returns heatmap
    ax5 = plt.subplot(3, 3, 5)
    monthly_returns = strategy_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
    monthly_table = monthly_returns.groupby([monthly_returns.index.year, monthly_returns.index.month]).first().unstack()
    sns.heatmap(monthly_table, annot=True, fmt='.2%', cmap='RdYlGn', center=0)
    plt.title('Monthly Returns Heatmap')
    
    # Rolling volatility
    ax6 = plt.subplot(3, 3, 6)
    rolling_vol = strategy_returns.rolling(63).std() * np.sqrt(252)
    plt.plot(rolling_vol.index, rolling_vol)
    plt.title('Rolling 3-Month Volatility')
    plt.grid(True, alpha=0.3)
    
    # Scatter plot vs benchmark
    ax7 = plt.subplot(3, 3, 7)
    plt.scatter(benchmark_returns, strategy_returns, alpha=0.5)
    z = np.polyfit(benchmark_returns, strategy_returns, 1)
    p = np.poly1d(z)
    plt.plot(benchmark_returns, p(benchmark_returns), "r--", alpha=0.8)
    plt.xlabel('Benchmark Returns')
    plt.ylabel('Strategy Returns')
    plt.title('Strategy vs Benchmark')
    plt.grid(True, alpha=0.3)
    
    # Best/Worst periods
    ax8 = plt.subplot(3, 3, 8)
    periods = ['Best Day', 'Worst Day', 'Best Month', 'Worst Month']
    values = [
        strategy_returns.max(),
        strategy_returns.min(),
        monthly_returns.max(),
        monthly_returns.min()
    ]
    colors = ['green', 'red', 'lightgreen', 'lightcoral']
    plt.bar(periods, values, color=colors)
    plt.title('Best/Worst Periods')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Performance metrics radar chart
    ax9 = plt.subplot(3, 3, 9, projection='polar')
    metrics = ['Sharpe', 'Sortino', 'Calmar', 'Information\nRatio']
    values = [
        calculate_sharpe_ratio(strategy_returns),
        calculate_sortino_ratio(strategy_returns),
        calculate_calmar_ratio(strategy_returns),
        calculate_information_ratio(strategy_returns, benchmark_returns)
    ]
    
    # Normalize values for radar chart (between 0 and 1)
    normalized_values = [(v - min(values)) / (max(values) - min(values)) for v in values]
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))
    normalized_values = normalized_values + [normalized_values[0]]
    
    ax9.plot(angles, normalized_values, 'o-', linewidth=2)
    ax9.fill(angles, normalized_values, alpha=0.25)
    ax9.set_xticks(angles[:-1])
    ax9.set_xticklabels(metrics)
    ax9.set_title('Performance Metrics Radar')
    
    plt.tight_layout()
    plt.show()

# Create performance dashboard
create_performance_dashboard(
    performance_data['Strategy_Returns'],
    performance_data['Benchmark_Returns']
)
```

## Practice Exercises

1. **Custom Metrics**: Develop your own performance metric combining return and risk
2. **Sector Attribution**: Implement detailed sector-level performance attribution
3. **Rolling Analysis**: Create rolling correlation analysis between strategy and factors
4. **Risk Budgeting**: Implement risk budgeting analysis for portfolio components
5. **Stress Testing**: Create stress test scenarios for different market conditions

## Key Takeaways

- Comprehensive performance analysis requires multiple metrics, not just returns
- Risk-adjusted measures provide better insight than absolute returns
- Drawdown analysis is crucial for understanding downside risk
- Attribution analysis helps identify sources of performance
- Rolling metrics show how performance changes over time
- Visualization is essential for communicating performance results

## Next Steps

This completes Module 2 on technical analysis and data processing. The next module would cover API integration and real-time data handling for live trading systems.
