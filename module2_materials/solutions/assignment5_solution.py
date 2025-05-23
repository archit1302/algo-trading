#!/usr/bin/env python3
"""
Module 2 - Assignment 5 Solution: Strategy Development and Backtesting
Comprehensive trading strategy development and backtesting framework.

Topics Covered:
- Trading strategy design and implementation
- Signal generation and filtering
- Position sizing and risk management
- Portfolio construction and management
- Backtesting framework and validation
- Strategy performance analysis
- Walk-forward analysis and optimization

Author: Financial Data Science Course
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class TradingStrategy:
    """
    Base class for trading strategies.
    """
    
    def __init__(self, name="Strategy"):
        """
        Initialize trading strategy.
        
        Parameters:
        name (str): Strategy name
        """
        self.name = name
        self.signals = None
        self.positions = None
        self.portfolio = None
        
    def generate_signals(self, data):
        """
        Generate trading signals. To be implemented by subclasses.
        
        Parameters:
        data (DataFrame): Financial data
        
        Returns:
        DataFrame: Signal data with buy/sell signals
        """
        raise NotImplementedError("Subclasses must implement generate_signals method")
    
    def apply_filters(self, signals, **kwargs):
        """
        Apply additional filters to signals.
        
        Parameters:
        signals (DataFrame): Raw signals
        **kwargs: Filter parameters
        
        Returns:
        DataFrame: Filtered signals
        """
        return signals

class MovingAverageCrossover(TradingStrategy):
    """
    Moving Average Crossover Strategy.
    """
    
    def __init__(self, short_window=20, long_window=50, name="MA_Crossover"):
        """
        Initialize MA crossover strategy.
        
        Parameters:
        short_window (int): Short moving average period
        long_window (int): Long moving average period
        name (str): Strategy name
        """
        super().__init__(name)
        self.short_window = short_window
        self.long_window = long_window
    
    def generate_signals(self, data):
        """Generate moving average crossover signals."""
        signals = pd.DataFrame(index=data.index)
        signals['price'] = data['close']
        signals['short_ma'] = data['close'].rolling(window=self.short_window).mean()
        signals['long_ma'] = data['close'].rolling(window=self.long_window).mean()
        
        # Generate signals
        signals['signal'] = 0
        signals['signal'][self.short_window:] = np.where(
            signals['short_ma'][self.short_window:] > signals['long_ma'][self.short_window:], 1, 0
        )
        signals['positions'] = signals['signal'].diff()
        
        return signals

class RSIMeanReversion(TradingStrategy):
    """
    RSI Mean Reversion Strategy.
    """
    
    def __init__(self, rsi_period=14, oversold=30, overbought=70, name="RSI_MeanReversion"):
        """
        Initialize RSI mean reversion strategy.
        
        Parameters:
        rsi_period (int): RSI calculation period
        oversold (int): Oversold threshold
        overbought (int): Overbought threshold
        name (str): Strategy name
        """
        super().__init__(name)
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
    
    def calculate_rsi(self, prices, period):
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def generate_signals(self, data):
        """Generate RSI mean reversion signals."""
        signals = pd.DataFrame(index=data.index)
        signals['price'] = data['close']
        signals['rsi'] = self.calculate_rsi(data['close'], self.rsi_period)
        
        # Generate signals
        signals['signal'] = 0
        signals.loc[signals['rsi'] < self.oversold, 'signal'] = 1  # Buy signal
        signals.loc[signals['rsi'] > self.overbought, 'signal'] = -1  # Sell signal
        signals['positions'] = signals['signal'].diff()
        
        return signals

class BollingerBandsStrategy(TradingStrategy):
    """
    Bollinger Bands Mean Reversion Strategy.
    """
    
    def __init__(self, period=20, std_dev=2, name="BollingerBands"):
        """
        Initialize Bollinger Bands strategy.
        
        Parameters:
        period (int): Moving average period
        std_dev (float): Standard deviation multiplier
        name (str): Strategy name
        """
        super().__init__(name)
        self.period = period
        self.std_dev = std_dev
    
    def generate_signals(self, data):
        """Generate Bollinger Bands signals."""
        signals = pd.DataFrame(index=data.index)
        signals['price'] = data['close']
        
        # Calculate Bollinger Bands
        signals['ma'] = data['close'].rolling(window=self.period).mean()
        signals['std'] = data['close'].rolling(window=self.period).std()
        signals['upper_band'] = signals['ma'] + (signals['std'] * self.std_dev)
        signals['lower_band'] = signals['ma'] - (signals['std'] * self.std_dev)
        
        # Generate signals
        signals['signal'] = 0
        signals.loc[data['close'] < signals['lower_band'], 'signal'] = 1  # Buy
        signals.loc[data['close'] > signals['upper_band'], 'signal'] = -1  # Sell
        signals['positions'] = signals['signal'].diff()
        
        return signals

class MomentumStrategy(TradingStrategy):
    """
    Price Momentum Strategy.
    """
    
    def __init__(self, lookback=20, threshold=0.02, name="Momentum"):
        """
        Initialize momentum strategy.
        
        Parameters:
        lookback (int): Momentum lookback period
        threshold (float): Momentum threshold for signal generation
        name (str): Strategy name
        """
        super().__init__(name)
        self.lookback = lookback
        self.threshold = threshold
    
    def generate_signals(self, data):
        """Generate momentum signals."""
        signals = pd.DataFrame(index=data.index)
        signals['price'] = data['close']
        
        # Calculate momentum
        signals['momentum'] = data['close'].pct_change(periods=self.lookback)
        
        # Generate signals
        signals['signal'] = 0
        signals.loc[signals['momentum'] > self.threshold, 'signal'] = 1  # Buy
        signals.loc[signals['momentum'] < -self.threshold, 'signal'] = -1  # Sell
        signals['positions'] = signals['signal'].diff()
        
        return signals

class BacktestEngine:
    """
    Comprehensive backtesting engine for trading strategies.
    """
    
    def __init__(self, initial_capital=100000, commission=0.001, slippage=0.0005):
        """
        Initialize backtesting engine.
        
        Parameters:
        initial_capital (float): Starting capital
        commission (float): Commission rate per trade
        slippage (float): Slippage rate per trade
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        
    def run_backtest(self, strategy, data, position_size=0.95, stop_loss=None, take_profit=None):
        """
        Run backtest for a given strategy.
        
        Parameters:
        strategy (TradingStrategy): Strategy to backtest
        data (DataFrame): Financial data
        position_size (float): Position size as fraction of portfolio
        stop_loss (float): Stop loss percentage
        take_profit (float): Take profit percentage
        
        Returns:
        dict: Backtest results
        """
        # Generate signals
        signals = strategy.generate_signals(data)
        
        # Initialize portfolio
        portfolio = pd.DataFrame(index=signals.index)
        portfolio['price'] = signals['price']
        portfolio['signal'] = signals['signal']
        portfolio['positions'] = signals['positions']
        
        # Calculate position sizes
        portfolio['holdings'] = 0.0
        portfolio['cash'] = self.initial_capital
        portfolio['total'] = self.initial_capital
        portfolio['returns'] = 0.0
        
        # Track trades
        trades = []
        current_position = 0
        entry_price = 0
        
        for i in range(1, len(portfolio)):
            date = portfolio.index[i]
            price = portfolio['price'].iloc[i]
            signal = portfolio['signal'].iloc[i]
            position_change = portfolio['positions'].iloc[i]
            
            # Copy previous values
            portfolio['cash'].iloc[i] = portfolio['cash'].iloc[i-1]
            portfolio['holdings'].iloc[i] = portfolio['holdings'].iloc[i-1]
            
            # Process position changes
            if position_change == 1:  # Buy signal
                if current_position <= 0:  # Enter long or close short
                    shares_to_buy = int((portfolio['cash'].iloc[i] * position_size) / price)
                    if shares_to_buy > 0:
                        cost = shares_to_buy * price * (1 + self.commission + self.slippage)
                        if cost <= portfolio['cash'].iloc[i]:
                            portfolio['cash'].iloc[i] -= cost
                            portfolio['holdings'].iloc[i] = shares_to_buy
                            current_position = shares_to_buy
                            entry_price = price
                            
                            trades.append({
                                'date': date,
                                'action': 'BUY',
                                'shares': shares_to_buy,
                                'price': price,
                                'value': cost
                            })
                            
            elif position_change == -1:  # Sell signal
                if current_position >= 0:  # Enter short or close long
                    if current_position > 0:  # Close long position
                        proceeds = current_position * price * (1 - self.commission - self.slippage)
                        portfolio['cash'].iloc[i] += proceeds
                        portfolio['holdings'].iloc[i] = 0
                        
                        trades.append({
                            'date': date,
                            'action': 'SELL',
                            'shares': current_position,
                            'price': price,
                            'value': proceeds
                        })
                        
                        current_position = 0
            
            # Apply stop loss and take profit
            if current_position > 0 and entry_price > 0:
                if stop_loss and price <= entry_price * (1 - stop_loss):
                    # Stop loss triggered
                    proceeds = current_position * price * (1 - self.commission - self.slippage)
                    portfolio['cash'].iloc[i] += proceeds
                    portfolio['holdings'].iloc[i] = 0
                    
                    trades.append({
                        'date': date,
                        'action': 'STOP_LOSS',
                        'shares': current_position,
                        'price': price,
                        'value': proceeds
                    })
                    
                    current_position = 0
                    
                elif take_profit and price >= entry_price * (1 + take_profit):
                    # Take profit triggered
                    proceeds = current_position * price * (1 - self.commission - self.slippage)
                    portfolio['cash'].iloc[i] += proceeds
                    portfolio['holdings'].iloc[i] = 0
                    
                    trades.append({
                        'date': date,
                        'action': 'TAKE_PROFIT',
                        'shares': current_position,
                        'price': price,
                        'value': proceeds
                    })
                    
                    current_position = 0
            
            # Calculate total portfolio value
            portfolio['total'].iloc[i] = portfolio['cash'].iloc[i] + (portfolio['holdings'].iloc[i] * price)
            portfolio['returns'].iloc[i] = portfolio['total'].iloc[i] / portfolio['total'].iloc[i-1] - 1
        
        # Calculate performance metrics
        results = self.calculate_performance_metrics(portfolio, trades)
        results['portfolio'] = portfolio
        results['trades'] = pd.DataFrame(trades) if trades else pd.DataFrame()
        results['signals'] = signals
        results['strategy_name'] = strategy.name
        
        return results
    
    def calculate_performance_metrics(self, portfolio, trades):
        """Calculate comprehensive performance metrics."""
        total_return = (portfolio['total'].iloc[-1] / portfolio['total'].iloc[0]) - 1
        annualized_return = (1 + total_return) ** (252 / len(portfolio)) - 1
        
        returns = portfolio['returns'].dropna()
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate and other trade statistics
        if trades:
            trade_df = pd.DataFrame(trades)
            winning_trades = len(trade_df[trade_df['action'].isin(['SELL', 'TAKE_PROFIT'])])
            total_trades = len(trade_df[trade_df['action'].isin(['SELL', 'STOP_LOSS', 'TAKE_PROFIT'])])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
        else:
            win_rate = 0
            total_trades = 0
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = annualized_return / downside_volatility if downside_volatility > 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'final_portfolio_value': portfolio['total'].iloc[-1]
        }
    
    def plot_backtest_results(self, results):
        """Plot comprehensive backtest results."""
        portfolio = results['portfolio']
        signals = results['signals']
        
        fig, axes = plt.subplots(3, 2, figsize=(20, 15))
        
        # Portfolio value over time
        axes[0, 0].plot(portfolio.index, portfolio['total'], label='Portfolio Value', linewidth=2)
        axes[0, 0].axhline(y=self.initial_capital, color='r', linestyle='--', alpha=0.7, label='Initial Capital')
        axes[0, 0].set_title(f'{results["strategy_name"]} - Portfolio Value')
        axes[0, 0].set_ylabel('Portfolio Value ($)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Price and signals
        axes[0, 1].plot(signals.index, signals['price'], label='Price', alpha=0.7)
        buy_signals = signals[signals['positions'] == 1]
        sell_signals = signals[signals['positions'] == -1]
        axes[0, 1].scatter(buy_signals.index, buy_signals['price'], 
                          color='green', marker='^', s=100, label='Buy', alpha=0.8)
        axes[0, 1].scatter(sell_signals.index, sell_signals['price'], 
                          color='red', marker='v', s=100, label='Sell', alpha=0.8)
        axes[0, 1].set_title('Price and Trading Signals')
        axes[0, 1].set_ylabel('Price ($)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Returns distribution
        returns = portfolio['returns'].dropna()
        axes[1, 0].hist(returns, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(returns.mean(), color='red', linestyle='--', label=f'Mean: {returns.mean():.4f}')
        axes[1, 0].set_title('Daily Returns Distribution')
        axes[1, 0].set_xlabel('Daily Returns')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        axes[1, 1].fill_between(portfolio.index[1:], drawdown, 0, alpha=0.7, color='red')
        axes[1, 1].set_title('Drawdown')
        axes[1, 1].set_ylabel('Drawdown (%)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Rolling Sharpe ratio
        rolling_returns = returns.rolling(window=252).mean() * 252
        rolling_vol = returns.rolling(window=252).std() * np.sqrt(252)
        rolling_sharpe = rolling_returns / rolling_vol
        axes[2, 0].plot(portfolio.index[252:], rolling_sharpe[252:], linewidth=2)
        axes[2, 0].axhline(y=1, color='green', linestyle='--', alpha=0.7, label='Sharpe = 1')
        axes[2, 0].set_title('Rolling 1-Year Sharpe Ratio')
        axes[2, 0].set_ylabel('Sharpe Ratio')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        # Performance metrics table
        metrics = {
            'Total Return': f"{results['total_return']:.2%}",
            'Annualized Return': f"{results['annualized_return']:.2%}",
            'Volatility': f"{results['volatility']:.2%}",
            'Sharpe Ratio': f"{results['sharpe_ratio']:.2f}",
            'Sortino Ratio': f"{results['sortino_ratio']:.2f}",
            'Calmar Ratio': f"{results['calmar_ratio']:.2f}",
            'Max Drawdown': f"{results['max_drawdown']:.2%}",
            'Win Rate': f"{results['win_rate']:.2%}",
            'Total Trades': f"{results['total_trades']}"
        }
        
        axes[2, 1].axis('tight')
        axes[2, 1].axis('off')
        table_data = [[key, value] for key, value in metrics.items()]
        table = axes[2, 1].table(cellText=table_data, colLabels=['Metric', 'Value'],
                                cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        axes[2, 1].set_title('Performance Metrics')
        
        plt.tight_layout()
        plt.show()
    
    def compare_strategies(self, strategy_results):
        """
        Compare multiple strategy results.
        
        Parameters:
        strategy_results (list): List of backtest results dictionaries
        """
        if len(strategy_results) < 2:
            print("Need at least 2 strategies to compare.")
            return
        
        # Create comparison DataFrame
        comparison_data = []
        for result in strategy_results:
            comparison_data.append({
                'Strategy': result['strategy_name'],
                'Total Return': result['total_return'],
                'Annualized Return': result['annualized_return'],
                'Volatility': result['volatility'],
                'Sharpe Ratio': result['sharpe_ratio'],
                'Sortino Ratio': result['sortino_ratio'],
                'Max Drawdown': result['max_drawdown'],
                'Win Rate': result['win_rate'],
                'Total Trades': result['total_trades']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Plot comparison
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Portfolio values
        axes[0, 0].set_title('Portfolio Value Comparison')
        for result in strategy_results:
            portfolio = result['portfolio']
            axes[0, 0].plot(portfolio.index, portfolio['total'], 
                           label=result['strategy_name'], linewidth=2)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylabel('Portfolio Value ($)')
        
        # Returns comparison
        metrics = ['Total Return', 'Annualized Return', 'Volatility']
        for i, metric in enumerate(metrics):
            ax = axes[0, i] if i == 0 else axes[0, 1] if i == 1 else axes[0, 2]
            if i > 0:  # Skip portfolio value plot
                comparison_df.plot(x='Strategy', y=metric, kind='bar', ax=ax, legend=False)
                ax.set_title(metric)
                ax.tick_params(axis='x', rotation=45)
        
        # Risk metrics
        risk_metrics = ['Sharpe Ratio', 'Max Drawdown', 'Win Rate']
        for i, metric in enumerate(risk_metrics):
            ax = axes[1, i]
            comparison_df.plot(x='Strategy', y=metric, kind='bar', ax=ax, legend=False)
            ax.set_title(metric)
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        # Print comparison table
        print("\n=== Strategy Comparison ===")
        print(comparison_df.round(4))
        
        return comparison_df

def generate_sample_data():
    """Generate sample financial data for backtesting."""
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='D')
    
    # Generate price with trend, volatility clustering, and mean reversion
    returns = []
    volatility = 0.02
    
    for i in range(len(dates)):
        # Volatility clustering
        volatility = 0.95 * volatility + 0.05 * 0.02 + 0.1 * np.random.normal(0, 0.001)
        volatility = max(0.005, min(0.05, volatility))  # Bound volatility
        
        # Return with mean reversion
        ret = np.random.normal(0.0005, volatility)  # Slight positive drift
        returns.append(ret)
    
    # Convert to prices
    price = 100 * np.exp(np.cumsum(returns))
    
    # Generate OHLCV data
    data = pd.DataFrame({
        'open': price * (1 + np.random.normal(0, 0.002, len(price))),
        'high': price * (1 + np.abs(np.random.normal(0.005, 0.003, len(price)))),
        'low': price * (1 - np.abs(np.random.normal(0.005, 0.003, len(price)))),
        'close': price,
        'volume': np.random.randint(500000, 2000000, len(price))
    }, index=dates)
    
    # Ensure OHLC logic
    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)
    
    return data

def main():
    """Main function demonstrating strategy development and backtesting."""
    print("=== Module 2 Assignment 5: Strategy Development and Backtesting ===\n")
    
    # Generate sample data
    print("1. Loading sample financial data...")
    data = generate_sample_data()
    print(f"Data period: {data.index[0].date()} to {data.index[-1].date()}")
    print(f"Total observations: {len(data)}")
    
    # Initialize backtesting engine
    backtest_engine = BacktestEngine(
        initial_capital=100000,
        commission=0.001,  # 0.1% commission
        slippage=0.0005    # 0.05% slippage
    )
    
    # Initialize strategies
    strategies = [
        MovingAverageCrossover(short_window=20, long_window=50),
        RSIMeanReversion(rsi_period=14, oversold=30, overbought=70),
        BollingerBandsStrategy(period=20, std_dev=2),
        MomentumStrategy(lookback=20, threshold=0.02)
    ]
    
    print(f"\n2. Testing {len(strategies)} strategies...")
    
    # Run backtests
    results = []
    for strategy in strategies:
        print(f"\nBacktesting {strategy.name}...")
        result = backtest_engine.run_backtest(
            strategy=strategy,
            data=data,
            position_size=0.95,
            stop_loss=0.05,  # 5% stop loss
            take_profit=0.10  # 10% take profit
        )
        results.append(result)
        
        # Print key metrics
        print(f"Total Return: {result['total_return']:.2%}")
        print(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {result['max_drawdown']:.2%}")
        print(f"Win Rate: {result['win_rate']:.2%}")
        print(f"Total Trades: {result['total_trades']}")
    
    # Plot individual strategy results
    print("\n3. Plotting individual strategy results...")
    for result in results:
        backtest_engine.plot_backtest_results(result)
    
    # Compare strategies
    print("\n4. Comparing strategies...")
    comparison_df = backtest_engine.compare_strategies(results)
    
    # Best strategy analysis
    best_sharpe = comparison_df.loc[comparison_df['Sharpe Ratio'].idxmax()]
    best_return = comparison_df.loc[comparison_df['Total Return'].idxmax()]
    
    print(f"\n5. Best Strategy Analysis:")
    print(f"Best Sharpe Ratio: {best_sharpe['Strategy']} ({best_sharpe['Sharpe Ratio']:.2f})")
    print(f"Best Total Return: {best_return['Strategy']} ({best_return['Total Return']:.2%})")
    
    # Risk analysis
    print(f"\n6. Risk Analysis:")
    for i, result in enumerate(results):
        strategy_name = result['strategy_name']
        portfolio = result['portfolio']
        returns = portfolio['returns'].dropna()
        
        # VaR calculation
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        
        print(f"\n{strategy_name}:")
        print(f"  Daily VaR (95%): {var_95:.4f}")
        print(f"  Daily VaR (99%): {var_99:.4f}")
        print(f"  Worst single day: {returns.min():.4f}")
        print(f"  Best single day: {returns.max():.4f}")
    
    print("\n=== Strategy Development and Backtesting Complete ===")
    print("\nThis analysis covers:")
    print("- Multiple trading strategy implementations")
    print("- Comprehensive backtesting framework")
    print("- Risk management with stop-loss and take-profit")
    print("- Performance metrics and comparison")
    print("- Risk analysis and Value-at-Risk calculations")

if __name__ == "__main__":
    main()
