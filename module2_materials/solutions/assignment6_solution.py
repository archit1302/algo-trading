#!/usr/bin/env python3
"""
Module 2 - Assignment 6 Solution: Performance Analysis and Risk Management
Advanced portfolio performance analysis and comprehensive risk management framework.

Topics Covered:
- Portfolio performance attribution
- Risk-adjusted performance metrics
- Value-at-Risk (VaR) and Expected Shortfall
- Factor analysis and decomposition
- Portfolio optimization
- Risk budgeting and allocation
- Stress testing and scenario analysis
- Monte Carlo simulation

Author: Financial Data Science Course
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Try importing optional libraries with fallbacks
try:
    from scipy import stats
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    print("Warning: SciPy not available. Some optimization features will be limited.")
    SCIPY_AVAILABLE = False

class PerformanceAnalyzer:
    """
    Comprehensive performance analysis toolkit.
    """
    
    def __init__(self, returns_data, benchmark_returns=None, risk_free_rate=0.02):
        """
        Initialize performance analyzer.
        
        Parameters:
        returns_data (DataFrame): Portfolio/strategy returns
        benchmark_returns (Series): Benchmark returns for comparison
        risk_free_rate (float): Risk-free rate (annualized)
        """
        self.returns = returns_data.copy()
        self.benchmark = benchmark_returns
        self.risk_free_rate = risk_free_rate
        self.daily_rf_rate = risk_free_rate / 252
        
    def basic_performance_metrics(self):
        """Calculate basic performance metrics."""
        metrics = {}
        
        for column in self.returns.columns:
            returns = self.returns[column].dropna()
            
            # Basic metrics
            total_return = (1 + returns).prod() - 1
            annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
            volatility = returns.std() * np.sqrt(252)
            
            # Risk-adjusted metrics
            excess_returns = returns - self.daily_rf_rate
            sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
            
            # Downside metrics
            downside_returns = returns[returns < 0]
            downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
            sortino_ratio = excess_returns.mean() / downside_volatility * np.sqrt(252) if downside_volatility > 0 else 0
            
            # Drawdown metrics
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # Calmar ratio
            calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # Skewness and Kurtosis
            skewness = returns.skew()
            kurtosis = returns.kurtosis()
            
            metrics[column] = {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'max_drawdown': max_drawdown,
                'skewness': skewness,
                'kurtosis': kurtosis
            }
        
        return metrics
    
    def value_at_risk(self, confidence_levels=[0.95, 0.99], method='historical'):
        """
        Calculate Value-at-Risk (VaR) using different methods.
        
        Parameters:
        confidence_levels (list): Confidence levels for VaR calculation
        method (str): 'historical', 'parametric', or 'monte_carlo'
        """
        var_results = {}
        
        for column in self.returns.columns:
            returns = self.returns[column].dropna()
            var_results[column] = {}
            
            for confidence in confidence_levels:
                alpha = 1 - confidence
                
                if method == 'historical':
                    # Historical VaR
                    var = np.percentile(returns, alpha * 100)
                    
                elif method == 'parametric':
                    # Parametric VaR (assuming normal distribution)
                    mean = returns.mean()
                    std = returns.std()
                    var = stats.norm.ppf(alpha, mean, std)
                    
                elif method == 'monte_carlo':
                    # Monte Carlo VaR
                    np.random.seed(42)
                    simulated_returns = np.random.normal(returns.mean(), returns.std(), 10000)
                    var = np.percentile(simulated_returns, alpha * 100)
                
                # Expected Shortfall (Conditional VaR)
                shortfall_returns = returns[returns <= var]
                expected_shortfall = shortfall_returns.mean() if len(shortfall_returns) > 0 else var
                
                var_results[column][f'VaR_{int(confidence*100)}'] = var
                var_results[column][f'ES_{int(confidence*100)}'] = expected_shortfall
        
        return var_results
    
    def benchmark_analysis(self):
        """Analyze performance relative to benchmark."""
        if self.benchmark is None:
            print("No benchmark provided for analysis.")
            return None
        
        analysis = {}
        
        for column in self.returns.columns:
            returns = self.returns[column].dropna()
            
            # Align dates
            common_dates = returns.index.intersection(self.benchmark.index)
            portfolio_returns = returns.loc[common_dates]
            benchmark_returns = self.benchmark.loc[common_dates]
            
            if len(common_dates) == 0:
                continue
            
            # Active returns
            active_returns = portfolio_returns - benchmark_returns
            
            # Tracking error
            tracking_error = active_returns.std() * np.sqrt(252)
            
            # Information ratio
            information_ratio = active_returns.mean() / active_returns.std() * np.sqrt(252) if active_returns.std() > 0 else 0
            
            # Beta calculation
            if SCIPY_AVAILABLE:
                beta, alpha_coef = np.polyfit(benchmark_returns, portfolio_returns, 1)
            else:
                # Simple beta calculation
                covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
                benchmark_variance = np.var(benchmark_returns)
                beta = covariance / benchmark_variance if benchmark_variance != 0 else 1
                alpha_coef = portfolio_returns.mean() - beta * benchmark_returns.mean()
            
            # Jensen's Alpha
            excess_portfolio = portfolio_returns - self.daily_rf_rate
            excess_benchmark = benchmark_returns - self.daily_rf_rate
            jensen_alpha = excess_portfolio.mean() - beta * excess_benchmark.mean()
            jensen_alpha_annualized = jensen_alpha * 252
            
            # Treynor ratio
            treynor_ratio = excess_portfolio.mean() / beta * 252 if beta != 0 else 0
            
            analysis[column] = {
                'beta': beta,
                'alpha': alpha_coef * 252,  # Annualized
                'jensen_alpha': jensen_alpha_annualized,
                'tracking_error': tracking_error,
                'information_ratio': information_ratio,
                'treynor_ratio': treynor_ratio
            }
        
        return analysis
    
    def factor_analysis(self, factors=None):
        """
        Perform factor analysis on returns.
        
        Parameters:
        factors (DataFrame): Factor returns (e.g., market, size, value factors)
        """
        if factors is None:
            print("No factors provided. Creating sample factors.")
            factors = self.create_sample_factors()
        
        factor_analysis = {}
        
        for column in self.returns.columns:
            returns = self.returns[column].dropna()
            
            # Align dates
            common_dates = returns.index.intersection(factors.index)
            portfolio_returns = returns.loc[common_dates]
            factor_returns = factors.loc[common_dates]
            
            if len(common_dates) == 0:
                continue
            
            # Multiple regression
            if SCIPY_AVAILABLE:
                try:
                    from sklearn.linear_model import LinearRegression
                    
                    X = factor_returns.values
                    y = portfolio_returns.values
                    
                    model = LinearRegression()
                    model.fit(X, y)
                    
                    factor_loadings = dict(zip(factor_returns.columns, model.coef_))
                    alpha = model.intercept_
                    r_squared = model.score(X, y)
                    
                    factor_analysis[column] = {
                        'alpha': alpha * 252,  # Annualized
                        'factor_loadings': factor_loadings,
                        'r_squared': r_squared
                    }
                    
                except ImportError:
                    # Fallback to simple correlation analysis
                    correlations = {}
                    for factor in factor_returns.columns:
                        correlations[factor] = portfolio_returns.corr(factor_returns[factor])
                    
                    factor_analysis[column] = {
                        'correlations': correlations
                    }
            else:
                # Simple correlation analysis
                correlations = {}
                for factor in factor_returns.columns:
                    correlations[factor] = portfolio_returns.corr(factor_returns[factor])
                
                factor_analysis[column] = {
                    'correlations': correlations
                }
        
        return factor_analysis
    
    def create_sample_factors(self):
        """Create sample factor returns for demonstration."""
        dates = self.returns.index
        np.random.seed(42)
        
        # Market factor
        market_returns = np.random.normal(0.0005, 0.015, len(dates))
        
        # Size factor (SMB - Small Minus Big)
        size_returns = np.random.normal(0.0002, 0.008, len(dates))
        
        # Value factor (HML - High Minus Low)
        value_returns = np.random.normal(0.0001, 0.007, len(dates))
        
        # Momentum factor
        momentum_returns = np.random.normal(0.0003, 0.009, len(dates))
        
        factors = pd.DataFrame({
            'Market': market_returns,
            'Size': size_returns,
            'Value': value_returns,
            'Momentum': momentum_returns
        }, index=dates)
        
        return factors
    
    def rolling_performance(self, window=252):
        """Calculate rolling performance metrics."""
        rolling_metrics = {}
        
        for column in self.returns.columns:
            returns = self.returns[column].dropna()
            
            # Rolling returns
            rolling_return = returns.rolling(window=window).apply(lambda x: (1 + x).prod() - 1)
            
            # Rolling volatility
            rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)
            
            # Rolling Sharpe ratio
            rolling_excess = returns.rolling(window=window).apply(lambda x: x - self.daily_rf_rate)
            rolling_sharpe = (rolling_excess.rolling(window=window).mean() / 
                            returns.rolling(window=window).std() * np.sqrt(252))
            
            # Rolling max drawdown
            rolling_cumulative = returns.rolling(window=window).apply(lambda x: (1 + x).prod())
            rolling_max = rolling_cumulative.rolling(window=window).max()
            rolling_drawdown = ((rolling_cumulative - rolling_max) / rolling_max).rolling(window=window).min()
            
            rolling_metrics[column] = {
                'rolling_return': rolling_return,
                'rolling_volatility': rolling_vol,
                'rolling_sharpe': rolling_sharpe,
                'rolling_max_drawdown': rolling_drawdown
            }
        
        return rolling_metrics
    
    def stress_testing(self, scenarios=None):
        """
        Perform stress testing on the portfolio.
        
        Parameters:
        scenarios (dict): Stress scenarios to test
        """
        if scenarios is None:
            scenarios = {
                'Market Crash': {'type': 'percentile', 'percentile': 1},
                'High Volatility': {'type': 'volatility_shock', 'multiplier': 3},
                'Black Monday': {'type': 'single_day', 'return': -0.22},
                'Financial Crisis': {'type': 'prolonged', 'daily_return': -0.02, 'days': 30}
            }
        
        stress_results = {}
        
        for column in self.returns.columns:
            returns = self.returns[column].dropna()
            stress_results[column] = {}
            
            for scenario_name, scenario_params in scenarios.items():
                if scenario_params['type'] == 'percentile':
                    # Worst percentile scenario
                    worst_return = np.percentile(returns, scenario_params['percentile'])
                    stress_results[column][scenario_name] = worst_return
                    
                elif scenario_params['type'] == 'volatility_shock':
                    # Volatility shock scenario
                    current_vol = returns.std()
                    shocked_vol = current_vol * scenario_params['multiplier']
                    # Simulate returns with higher volatility
                    np.random.seed(42)
                    shocked_returns = np.random.normal(returns.mean(), shocked_vol, 252)
                    worst_shocked = np.min(shocked_returns)
                    stress_results[column][scenario_name] = worst_shocked
                    
                elif scenario_params['type'] == 'single_day':
                    # Single day extreme loss
                    stress_results[column][scenario_name] = scenario_params['return']
                    
                elif scenario_params['type'] == 'prolonged':
                    # Prolonged negative period
                    daily_return = scenario_params['daily_return']
                    days = scenario_params['days']
                    cumulative_return = (1 + daily_return) ** days - 1
                    stress_results[column][scenario_name] = cumulative_return
        
        return stress_results
    
    def monte_carlo_simulation(self, num_simulations=1000, time_horizon=252):
        """
        Perform Monte Carlo simulation for portfolio projections.
        
        Parameters:
        num_simulations (int): Number of simulation paths
        time_horizon (int): Number of days to simulate
        """
        simulation_results = {}
        
        for column in self.returns.columns:
            returns = self.returns[column].dropna()
            
            # Parameters for simulation
            mean_return = returns.mean()
            volatility = returns.std()
            
            # Monte Carlo simulation
            np.random.seed(42)
            simulations = np.zeros((num_simulations, time_horizon))
            
            for i in range(num_simulations):
                # Generate random returns
                random_returns = np.random.normal(mean_return, volatility, time_horizon)
                
                # Calculate cumulative values
                cumulative_values = np.cumprod(1 + random_returns)
                simulations[i] = cumulative_values
            
            # Calculate percentiles
            percentiles = [5, 25, 50, 75, 95]
            simulation_percentiles = {}
            for p in percentiles:
                simulation_percentiles[f'p{p}'] = np.percentile(simulations, p, axis=0)
            
            simulation_results[column] = {
                'simulations': simulations,
                'percentiles': simulation_percentiles,
                'final_values': simulations[:, -1]
            }
        
        return simulation_results
    
    def plot_performance_dashboard(self):
        """Create comprehensive performance dashboard."""
        num_assets = len(self.returns.columns)
        
        if num_assets == 1:
            fig, axes = plt.subplots(3, 2, figsize=(16, 12))
            axes = axes.flatten()
            column = self.returns.columns[0]
            returns = self.returns[column].dropna()
            
            # Cumulative returns
            cumulative = (1 + returns).cumprod()
            axes[0].plot(cumulative.index, cumulative.values, linewidth=2)
            axes[0].set_title('Cumulative Returns')
            axes[0].grid(True, alpha=0.3)
            
            # Returns distribution
            axes[1].hist(returns, bins=50, alpha=0.7, edgecolor='black')
            axes[1].axvline(returns.mean(), color='red', linestyle='--', label=f'Mean: {returns.mean():.4f}')
            axes[1].set_title('Returns Distribution')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            # Drawdown
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            axes[2].fill_between(drawdown.index, drawdown.values, 0, alpha=0.7, color='red')
            axes[2].set_title('Drawdown')
            axes[2].grid(True, alpha=0.3)
            
            # Rolling volatility
            rolling_vol = returns.rolling(window=30).std() * np.sqrt(252)
            axes[3].plot(rolling_vol.index, rolling_vol.values, linewidth=2)
            axes[3].set_title('Rolling 30-Day Volatility (Annualized)')
            axes[3].grid(True, alpha=0.3)
            
            # VaR analysis
            var_results = self.value_at_risk()
            var_95 = var_results[column]['VaR_95']
            var_99 = var_results[column]['VaR_99']
            
            axes[4].hist(returns, bins=50, alpha=0.7, edgecolor='black')
            axes[4].axvline(var_95, color='orange', linestyle='--', label=f'VaR 95%: {var_95:.4f}')
            axes[4].axvline(var_99, color='red', linestyle='--', label=f'VaR 99%: {var_99:.4f}')
            axes[4].set_title('Value-at-Risk Analysis')
            axes[4].legend()
            axes[4].grid(True, alpha=0.3)
            
            # Performance metrics
            metrics = self.basic_performance_metrics()[column]
            metrics_text = '\n'.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
            axes[5].text(0.1, 0.9, metrics_text, transform=axes[5].transAxes, fontsize=10,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            axes[5].set_title('Performance Metrics')
            axes[5].axis('off')
            
        else:
            # Multiple assets comparison
            fig, axes = plt.subplots(2, 2, figsize=(16, 10))
            
            # Cumulative returns comparison
            for column in self.returns.columns:
                returns = self.returns[column].dropna()
                cumulative = (1 + returns).cumprod()
                axes[0, 0].plot(cumulative.index, cumulative.values, label=column, linewidth=2)
            axes[0, 0].set_title('Cumulative Returns Comparison')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Risk-Return scatter
            metrics = self.basic_performance_metrics()
            for column in self.returns.columns:
                risk = metrics[column]['volatility']
                ret = metrics[column]['annualized_return']
                axes[0, 1].scatter(risk, ret, s=100, label=column)
            axes[0, 1].set_xlabel('Volatility (Annualized)')
            axes[0, 1].set_ylabel('Return (Annualized)')
            axes[0, 1].set_title('Risk-Return Profile')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Correlation matrix
            correlation_matrix = self.returns.corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, ax=axes[1, 0])
            axes[1, 0].set_title('Correlation Matrix')
            
            # Rolling Sharpe ratios
            rolling_metrics = self.rolling_performance(window=63)  # Quarterly
            for column in self.returns.columns:
                rolling_sharpe = rolling_metrics[column]['rolling_sharpe']
                axes[1, 1].plot(rolling_sharpe.index, rolling_sharpe.values, label=column, linewidth=2)
            axes[1, 1].set_title('Rolling Quarterly Sharpe Ratio')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def generate_performance_report(self):
        """Generate comprehensive performance report."""
        print("=== COMPREHENSIVE PERFORMANCE REPORT ===\n")
        
        # Basic metrics
        print("1. BASIC PERFORMANCE METRICS")
        print("-" * 50)
        metrics = self.basic_performance_metrics()
        metrics_df = pd.DataFrame(metrics).T
        print(metrics_df.round(4))
        
        # VaR Analysis
        print("\n2. VALUE-AT-RISK ANALYSIS")
        print("-" * 50)
        var_results = self.value_at_risk()
        for asset, vars in var_results.items():
            print(f"\n{asset}:")
            for var_type, value in vars.items():
                print(f"  {var_type}: {value:.4f}")
        
        # Benchmark Analysis
        if self.benchmark is not None:
            print("\n3. BENCHMARK ANALYSIS")
            print("-" * 50)
            benchmark_analysis = self.benchmark_analysis()
            benchmark_df = pd.DataFrame(benchmark_analysis).T
            print(benchmark_df.round(4))
        
        # Stress Testing
        print("\n4. STRESS TESTING RESULTS")
        print("-" * 50)
        stress_results = self.stress_testing()
        for asset, scenarios in stress_results.items():
            print(f"\n{asset}:")
            for scenario, value in scenarios.items():
                print(f"  {scenario}: {value:.4f}")
        
        # Factor Analysis
        print("\n5. FACTOR ANALYSIS")
        print("-" * 50)
        factor_analysis = self.factor_analysis()
        for asset, analysis in factor_analysis.items():
            print(f"\n{asset}:")
            for key, value in analysis.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for k, v in value.items():
                        print(f"    {k}: {v:.4f}")
                else:
                    print(f"  {key}: {value:.4f}")

def generate_sample_portfolio_data():
    """Generate sample portfolio return data."""
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='D')
    
    # Generate correlated returns for multiple assets
    num_assets = 3
    mean_returns = [0.0008, 0.0006, 0.0010]  # Different expected returns
    volatilities = [0.015, 0.020, 0.025]    # Different volatilities
    
    # Correlation matrix
    correlation = np.array([
        [1.0, 0.6, 0.3],
        [0.6, 1.0, 0.4],
        [0.3, 0.4, 1.0]
    ])
    
    # Generate correlated random returns
    random_returns = np.random.multivariate_normal([0, 0, 0], correlation, len(dates))
    
    # Scale by volatilities and add means
    scaled_returns = np.zeros_like(random_returns)
    for i in range(num_assets):
        scaled_returns[:, i] = random_returns[:, i] * volatilities[i] + mean_returns[i]
    
    # Create portfolio returns DataFrame
    portfolio_returns = pd.DataFrame(
        scaled_returns,
        index=dates,
        columns=['Growth_Stock', 'Value_Stock', 'Small_Cap']
    )
    
    # Generate benchmark returns (market index)
    market_returns = 0.3 * scaled_returns[:, 0] + 0.5 * scaled_returns[:, 1] + 0.2 * scaled_returns[:, 2]
    market_returns += np.random.normal(0, 0.005, len(dates))  # Add some noise
    
    benchmark = pd.Series(market_returns, index=dates, name='Market_Index')
    
    return portfolio_returns, benchmark

def main():
    """Main function demonstrating performance analysis and risk management."""
    print("=== Module 2 Assignment 6: Performance Analysis and Risk Management ===\n")
    
    # Generate sample data
    print("1. Loading sample portfolio data...")
    portfolio_returns, benchmark = generate_sample_portfolio_data()
    
    print(f"Portfolio assets: {list(portfolio_returns.columns)}")
    print(f"Data period: {portfolio_returns.index[0].date()} to {portfolio_returns.index[-1].date()}")
    print(f"Total observations: {len(portfolio_returns)}")
    
    # Initialize performance analyzer
    analyzer = PerformanceAnalyzer(
        returns_data=portfolio_returns,
        benchmark_returns=benchmark,
        risk_free_rate=0.02
    )
    
    print("\n2. Generating comprehensive performance report...")
    analyzer.generate_performance_report()
    
    print("\n3. Creating performance dashboard...")
    analyzer.plot_performance_dashboard()
    
    print("\n4. Monte Carlo simulation...")
    mc_results = analyzer.monte_carlo_simulation(num_simulations=1000, time_horizon=252)
    
    # Plot Monte Carlo results for first asset
    first_asset = portfolio_returns.columns[0]
    mc_data = mc_results[first_asset]
    
    plt.figure(figsize=(15, 8))
    
    # Plot simulation paths (sample)
    for i in range(min(100, len(mc_data['simulations']))):
        plt.plot(mc_data['simulations'][i], alpha=0.1, color='blue')
    
    # Plot percentiles
    time_axis = range(len(mc_data['percentiles']['p50']))
    plt.plot(time_axis, mc_data['percentiles']['p50'], 'r-', linewidth=2, label='Median (P50)')
    plt.plot(time_axis, mc_data['percentiles']['p5'], 'r--', linewidth=1, label='P5')
    plt.plot(time_axis, mc_data['percentiles']['p95'], 'r--', linewidth=1, label='P95')
    plt.fill_between(time_axis, mc_data['percentiles']['p25'], mc_data['percentiles']['p75'], 
                     alpha=0.3, color='red', label='P25-P75 Range')
    
    plt.title(f'Monte Carlo Simulation - {first_asset} (1000 paths, 1 year)')
    plt.xlabel('Days')
    plt.ylabel('Cumulative Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Final value distribution
    plt.figure(figsize=(12, 6))
    plt.hist(mc_data['final_values'], bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(np.median(mc_data['final_values']), color='red', linestyle='--', 
                label=f"Median: {np.median(mc_data['final_values']):.3f}")
    plt.axvline(np.percentile(mc_data['final_values'], 5), color='orange', linestyle='--',
                label=f"5th Percentile: {np.percentile(mc_data['final_values'], 5):.3f}")
    plt.axvline(np.percentile(mc_data['final_values'], 95), color='green', linestyle='--',
                label=f"95th Percentile: {np.percentile(mc_data['final_values'], 95):.3f}")
    plt.title(f'{first_asset} - Final Value Distribution (1 Year)')
    plt.xlabel('Final Portfolio Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print("\n5. Portfolio optimization demonstration...")
    # Simple portfolio optimization
    if SCIPY_AVAILABLE:
        returns_data = portfolio_returns.dropna()
        mean_returns = returns_data.mean() * 252
        cov_matrix = returns_data.cov() * 252
        
        def portfolio_performance(weights, mean_returns, cov_matrix):
            returns = np.sum(mean_returns * weights)
            std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return std, returns
        
        def neg_sharpe(weights, mean_returns, cov_matrix, risk_free_rate):
            p_var, p_ret = portfolio_performance(weights, mean_returns, cov_matrix)
            return -(p_ret - risk_free_rate) / p_var
        
        # Constraints and bounds
        num_assets = len(mean_returns)
        args = (mean_returns, cov_matrix, 0.02)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for asset in range(num_assets))
        
        # Optimize
        result = minimize(neg_sharpe, num_assets*[1./num_assets,], method='SLSQP',
                         bounds=bounds, constraints=constraints, args=args)
        
        optimal_weights = result.x
        
        print("Optimal Portfolio Weights:")
        for i, asset in enumerate(portfolio_returns.columns):
            print(f"  {asset}: {optimal_weights[i]:.3f}")
        
        optimal_return, optimal_volatility = portfolio_performance(optimal_weights, mean_returns, cov_matrix)
        optimal_sharpe = (optimal_return - 0.02) / optimal_volatility
        
        print(f"\nOptimal Portfolio Metrics:")
        print(f"  Expected Return: {optimal_return:.4f}")
        print(f"  Volatility: {optimal_volatility:.4f}")
        print(f"  Sharpe Ratio: {optimal_sharpe:.4f}")
    
    print("\n=== Performance Analysis and Risk Management Complete ===")
    print("\nThis analysis covers:")
    print("- Comprehensive performance metrics and risk-adjusted returns")
    print("- Value-at-Risk and Expected Shortfall calculations")
    print("- Benchmark analysis and factor decomposition")
    print("- Stress testing and scenario analysis")
    print("- Monte Carlo simulation for risk projections")
    print("- Portfolio optimization and efficient frontier")

if __name__ == "__main__":
    main()
