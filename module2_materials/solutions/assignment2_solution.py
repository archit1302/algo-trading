#!/usr/bin/env python3
"""
Assignment 2 Solution: Technical Indicators Implementation
Module 2: Technical Analysis and Data Processing

This solution implements comprehensive technical indicators including RSI, MACD, 
Bollinger Bands, and custom indicators with signal generation and backtesting.

Author: Financial Analytics Course
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class TechnicalIndicators:
    """
    Comprehensive technical indicators implementation for financial analysis.
    """
    
    def __init__(self, data):
        """
        Initialize with OHLCV data.
        
        Args:
            data (pd.DataFrame): Stock data with OHLCV columns
        """
        self.data = data.copy()
        self.signals = pd.DataFrame(index=data.index)
    
    def rsi(self, period=14, column='Close'):
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            period (int): RSI calculation period
            column (str): Price column to use
            
        Returns:
            pd.Series: RSI values
        """
        delta = self.data[column].diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        # Calculate average gains and losses using Wilder's smoothing
        avg_gains = gains.ewm(alpha=1/period).mean()
        avg_losses = losses.ewm(alpha=1/period).mean()
        
        # Calculate RS and RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        self.data[f'RSI_{period}'] = rsi
        return rsi
    
    def macd(self, fast_period=12, slow_period=26, signal_period=9, column='Close'):
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            fast_period (int): Fast EMA period
            slow_period (int): Slow EMA period
            signal_period (int): Signal line EMA period
            column (str): Price column to use
            
        Returns:
            tuple: MACD line, Signal line, Histogram
        """
        # Calculate EMAs
        ema_fast = self.data[column].ewm(span=fast_period).mean()
        ema_slow = self.data[column].ewm(span=slow_period).mean()
        
        # MACD line
        macd_line = ema_fast - ema_slow
        
        # Signal line
        signal_line = macd_line.ewm(span=signal_period).mean()
        
        # Histogram
        histogram = macd_line - signal_line
        
        # Store in dataframe
        self.data['MACD'] = macd_line
        self.data['MACD_Signal'] = signal_line
        self.data['MACD_Histogram'] = histogram
        
        return macd_line, signal_line, histogram
    
    def bollinger_bands(self, period=20, std_dev=2, column='Close'):
        """
        Calculate Bollinger Bands.
        
        Args:
            period (int): Moving average period
            std_dev (float): Standard deviation multiplier
            column (str): Price column to use
            
        Returns:
            tuple: Upper band, Middle band (SMA), Lower band
        """
        # Calculate SMA and standard deviation
        sma = self.data[column].rolling(window=period).mean()
        std = self.data[column].rolling(window=period).std()
        
        # Calculate bands
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        # Store in dataframe
        self.data[f'BB_Upper_{period}'] = upper_band
        self.data[f'BB_Middle_{period}'] = sma
        self.data[f'BB_Lower_{period}'] = lower_band
        
        # Calculate %B (position within bands)
        self.data[f'BB_Percent_B'] = ((self.data[column] - lower_band) / 
                                     (upper_band - lower_band)) * 100
        
        # Calculate Bandwidth
        self.data[f'BB_Bandwidth'] = ((upper_band - lower_band) / sma) * 100
        
        return upper_band, sma, lower_band
    
    def stochastic_oscillator(self, k_period=14, d_period=3):
        """
        Calculate Stochastic Oscillator (%K and %D).
        
        Args:
            k_period (int): %K calculation period
            d_period (int): %D smoothing period
            
        Returns:
            tuple: %K, %D
        """
        # Get lowest low and highest high over the period
        lowest_low = self.data['Low'].rolling(window=k_period).min()
        highest_high = self.data['High'].rolling(window=k_period).max()
        
        # Calculate %K
        k_percent = ((self.data['Close'] - lowest_low) / 
                    (highest_high - lowest_low)) * 100
        
        # Calculate %D (smoothed %K)
        d_percent = k_percent.rolling(window=d_period).mean()
        
        self.data['Stoch_K'] = k_percent
        self.data['Stoch_D'] = d_percent
        
        return k_percent, d_percent
    
    def atr(self, period=14):
        """
        Calculate Average True Range (ATR).
        
        Args:
            period (int): ATR calculation period
            
        Returns:
            pd.Series: ATR values
        """
        # Calculate True Range
        high_low = self.data['High'] - self.data['Low']
        high_close_prev = np.abs(self.data['High'] - self.data['Close'].shift(1))
        low_close_prev = np.abs(self.data['Low'] - self.data['Close'].shift(1))
        
        true_range = np.maximum.reduce([high_low, high_close_prev, low_close_prev])
        
        # Calculate ATR using Wilder's smoothing
        atr = true_range.ewm(alpha=1/period).mean()
        
        self.data[f'ATR_{period}'] = atr
        return atr
    
    def williams_r(self, period=14):
        """
        Calculate Williams %R.
        
        Args:
            period (int): Calculation period
            
        Returns:
            pd.Series: Williams %R values
        """
        highest_high = self.data['High'].rolling(window=period).max()
        lowest_low = self.data['Low'].rolling(window=period).min()
        
        williams_r = ((highest_high - self.data['Close']) / 
                     (highest_high - lowest_low)) * -100
        
        self.data[f'Williams_R_{period}'] = williams_r
        return williams_r
    
    def momentum(self, period=10, column='Close'):
        """
        Calculate Price Momentum.
        
        Args:
            period (int): Momentum calculation period
            column (str): Price column to use
            
        Returns:
            pd.Series: Momentum values
        """
        momentum = self.data[column] / self.data[column].shift(period) - 1
        self.data[f'Momentum_{period}'] = momentum * 100
        return momentum * 100
    
    def custom_trend_strength(self, short_period=10, long_period=30):
        """
        Custom indicator: Trend Strength Index.
        Combines multiple timeframes to assess trend strength.
        
        Args:
            short_period (int): Short-term period
            long_period (int): Long-term period
            
        Returns:
            pd.Series: Trend Strength values (0-100)
        """
        # Component 1: Price vs Moving Averages
        sma_short = self.data['Close'].rolling(window=short_period).mean()
        sma_long = self.data['Close'].rolling(window=long_period).mean()
        
        price_vs_short = ((self.data['Close'] - sma_short) / sma_short) * 100
        price_vs_long = ((self.data['Close'] - sma_long) / sma_long) * 100
        
        # Component 2: Moving Average Slope
        ma_slope_short = sma_short.pct_change(5) * 100
        ma_slope_long = sma_long.pct_change(10) * 100
        
        # Component 3: Volume Trend
        volume_ma = self.data['Volume'].rolling(window=short_period).mean()
        volume_trend = ((self.data['Volume'] - volume_ma) / volume_ma) * 100
        
        # Combine components with weights
        trend_strength = (
            price_vs_short * 0.3 +
            price_vs_long * 0.3 +
            ma_slope_short * 0.2 +
            ma_slope_long * 0.1 +
            volume_trend * 0.1
        )
        
        # Normalize to 0-100 scale
        trend_strength_normalized = ((trend_strength + 50) / 100) * 100
        trend_strength_normalized = np.clip(trend_strength_normalized, 0, 100)
        
        self.data['Trend_Strength'] = trend_strength_normalized
        return trend_strength_normalized

class SignalGenerator:
    """
    Generate trading signals based on technical indicators.
    """
    
    def __init__(self, data):
        """
        Initialize with data containing technical indicators.
        
        Args:
            data (pd.DataFrame): Data with technical indicators
        """
        self.data = data.copy()
        self.signals = pd.DataFrame(index=data.index)
    
    def rsi_signals(self, rsi_column='RSI_14', oversold=30, overbought=70):
        """
        Generate RSI-based trading signals.
        
        Args:
            rsi_column (str): RSI column name
            oversold (float): Oversold threshold
            overbought (float): Overbought threshold
            
        Returns:
            pd.Series: Trading signals (-1, 0, 1)
        """
        signals = pd.Series(0, index=self.data.index)
        
        # Buy signals (RSI crosses above oversold)
        buy_condition = (
            (self.data[rsi_column] > oversold) & 
            (self.data[rsi_column].shift(1) <= oversold)
        )
        
        # Sell signals (RSI crosses below overbought)
        sell_condition = (
            (self.data[rsi_column] < overbought) & 
            (self.data[rsi_column].shift(1) >= overbought)
        )
        
        signals[buy_condition] = 1
        signals[sell_condition] = -1
        
        self.signals['RSI_Signal'] = signals
        return signals
    
    def macd_signals(self):
        """
        Generate MACD-based trading signals.
        
        Returns:
            pd.Series: Trading signals (-1, 0, 1)
        """
        signals = pd.Series(0, index=self.data.index)
        
        # Buy signal: MACD crosses above signal line
        buy_condition = (
            (self.data['MACD'] > self.data['MACD_Signal']) &
            (self.data['MACD'].shift(1) <= self.data['MACD_Signal'].shift(1))
        )
        
        # Sell signal: MACD crosses below signal line
        sell_condition = (
            (self.data['MACD'] < self.data['MACD_Signal']) &
            (self.data['MACD'].shift(1) >= self.data['MACD_Signal'].shift(1))
        )
        
        signals[buy_condition] = 1
        signals[sell_condition] = -1
        
        self.signals['MACD_Signal'] = signals
        return signals
    
    def bollinger_signals(self, bb_period=20):
        """
        Generate Bollinger Bands-based trading signals.
        
        Args:
            bb_period (int): Bollinger Bands period
            
        Returns:
            pd.Series: Trading signals (-1, 0, 1)
        """
        signals = pd.Series(0, index=self.data.index)
        
        upper_col = f'BB_Upper_{bb_period}'
        lower_col = f'BB_Lower_{bb_period}'
        
        # Buy signal: Price touches lower band and bounces
        buy_condition = (
            (self.data['Close'] <= self.data[lower_col]) &
            (self.data['Close'].shift(1) > self.data[lower_col].shift(1))
        )
        
        # Sell signal: Price touches upper band
        sell_condition = (
            (self.data['Close'] >= self.data[upper_col]) &
            (self.data['Close'].shift(1) < self.data[upper_col].shift(1))
        )
        
        signals[buy_condition] = 1
        signals[sell_condition] = -1
        
        self.signals['BB_Signal'] = signals
        return signals
    
    def composite_signals(self, weights=None):
        """
        Generate composite signals combining multiple indicators.
        
        Args:
            weights (dict): Weights for different signals
            
        Returns:
            pd.Series: Composite trading signals
        """
        if weights is None:
            weights = {
                'RSI_Signal': 0.3,
                'MACD_Signal': 0.4,
                'BB_Signal': 0.3
            }
        
        composite = pd.Series(0.0, index=self.data.index)
        
        for signal_name, weight in weights.items():
            if signal_name in self.signals.columns:
                composite += self.signals[signal_name] * weight
        
        # Convert to discrete signals
        composite_signals = pd.Series(0, index=self.data.index)
        composite_signals[composite > 0.5] = 1
        composite_signals[composite < -0.5] = -1
        
        self.signals['Composite_Signal'] = composite_signals
        return composite_signals

class IndicatorBacktester:
    """
    Backtest technical indicator strategies.
    """
    
    def __init__(self, data, initial_capital=100000):
        """
        Initialize backtester.
        
        Args:
            data (pd.DataFrame): Data with signals
            initial_capital (float): Starting capital
        """
        self.data = data.copy()
        self.initial_capital = initial_capital
        self.results = {}
    
    def backtest_strategy(self, signal_column, transaction_cost=0.001):
        """
        Backtest a signal-based strategy.
        
        Args:
            signal_column (str): Column containing trading signals
            transaction_cost (float): Transaction cost per trade
            
        Returns:
            dict: Backtest results
        """
        signals = self.data[signal_column]
        prices = self.data['Close']
        
        # Initialize tracking variables
        position = 0
        cash = self.initial_capital
        holdings = 0
        portfolio_values = []
        trades = []
        
        for i, (date, signal) in enumerate(signals.items()):
            current_price = prices.loc[date]
            
            # Calculate current portfolio value
            portfolio_value = cash + (holdings * current_price)
            portfolio_values.append(portfolio_value)
            
            # Execute trades based on signals
            if signal == 1 and position == 0:  # Buy signal
                # Calculate shares to buy (use all available cash)
                shares_to_buy = int(cash / (current_price * (1 + transaction_cost)))
                if shares_to_buy > 0:
                    cost = shares_to_buy * current_price * (1 + transaction_cost)
                    cash -= cost
                    holdings += shares_to_buy
                    position = 1
                    
                    trades.append({
                        'date': date,
                        'action': 'BUY',
                        'shares': shares_to_buy,
                        'price': current_price,
                        'cost': cost
                    })
            
            elif signal == -1 and position == 1:  # Sell signal
                if holdings > 0:
                    proceeds = holdings * current_price * (1 - transaction_cost)
                    cash += proceeds
                    
                    trades.append({
                        'date': date,
                        'action': 'SELL',
                        'shares': holdings,
                        'price': current_price,
                        'proceeds': proceeds
                    })
                    
                    holdings = 0
                    position = 0
        
        # Final portfolio value
        final_value = cash + (holdings * prices.iloc[-1])
        portfolio_values.append(final_value)
        
        # Calculate performance metrics
        portfolio_series = pd.Series(portfolio_values, index=signals.index)
        returns = portfolio_series.pct_change().dropna()
        
        total_return = (final_value / self.initial_capital - 1) * 100
        annualized_return = (final_value / self.initial_capital) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = (annualized_return - 0.06) / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        peak = portfolio_series.expanding().max()
        drawdown = (portfolio_series - peak) / peak
        max_drawdown = drawdown.min() * 100
        
        # Win rate
        if len(trades) >= 2:
            trade_returns = []
            for i in range(1, len(trades), 2):  # Every sell trade
                if i < len(trades) and trades[i]['action'] == 'SELL':
                    buy_price = trades[i-1]['price']
                    sell_price = trades[i]['price']
                    trade_return = (sell_price / buy_price - 1) * 100
                    trade_returns.append(trade_return)
            
            if trade_returns:
                win_rate = sum(1 for ret in trade_returns if ret > 0) / len(trade_returns) * 100
                avg_win = np.mean([ret for ret in trade_returns if ret > 0]) if any(ret > 0 for ret in trade_returns) else 0
                avg_loss = np.mean([ret for ret in trade_returns if ret < 0]) if any(ret < 0 for ret in trade_returns) else 0
            else:
                win_rate = 0
                avg_win = 0
                avg_loss = 0
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
        
        results = {
            'strategy': signal_column,
            'total_return': total_return,
            'annualized_return': annualized_return * 100,
            'volatility': volatility * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': len(trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'final_value': final_value,
            'portfolio_values': portfolio_series,
            'trades': trades
        }
        
        self.results[signal_column] = results
        return results

def create_technical_dashboard(data, symbol="Stock"):
    """
    Create comprehensive technical analysis dashboard.
    
    Args:
        data (pd.DataFrame): Data with technical indicators
        symbol (str): Stock symbol for titles
    """
    # Create subplots
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[
            f'{symbol} - Price with Bollinger Bands',
            'RSI (14)',
            'MACD',
            'Volume'
        ],
        row_heights=[0.4, 0.2, 0.2, 0.2]
    )
    
    # Price chart with Bollinger Bands
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # Bollinger Bands
    if 'BB_Upper_20' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['BB_Upper_20'],
                name='BB Upper',
                line=dict(color='red', width=1),
                opacity=0.7
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['BB_Lower_20'],
                name='BB Lower',
                line=dict(color='red', width=1),
                fill='tonexty',
                fillcolor='rgba(255,0,0,0.1)',
                opacity=0.7
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['BB_Middle_20'],
                name='BB Middle',
                line=dict(color='blue', width=1, dash='dash')
            ),
            row=1, col=1
        )
    
    # RSI
    if 'RSI_14' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['RSI_14'],
                name='RSI',
                line=dict(color='purple')
            ),
            row=2, col=1
        )
        
        # RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", row=2, col=1)
    
    # MACD
    if 'MACD' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['MACD'],
                name='MACD',
                line=dict(color='blue')
            ),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['MACD_Signal'],
                name='Signal',
                line=dict(color='red')
            ),
            row=3, col=1
        )
        
        # MACD Histogram
        colors = ['green' if val >= 0 else 'red' for val in data['MACD_Histogram']]
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['MACD_Histogram'],
                name='Histogram',
                marker_color=colors,
                opacity=0.6
            ),
            row=3, col=1
        )
    
    # Volume
    fig.add_trace(
        go.Bar(
            x=data.index,
            y=data['Volume'],
            name='Volume',
            marker_color='lightblue',
            opacity=0.7
        ),
        row=4, col=1
    )
    
    # Update layout
    fig.update_layout(
        title=f'{symbol} - Technical Analysis Dashboard',
        xaxis_rangeslider_visible=False,
        height=1000,
        showlegend=True
    )
    
    # Update y-axis labels
    fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    fig.update_yaxes(title_text="Volume", row=4, col=1)
    
    fig.show()
    return fig

def generate_technical_report(results):
    """
    Generate comprehensive technical analysis report.
    
    Args:
        results (dict): Backtest results for multiple strategies
        
    Returns:
        str: Formatted report
    """
    report = f"""
{'='*80}
TECHNICAL INDICATORS ANALYSIS REPORT
{'='*80}

STRATEGY PERFORMANCE COMPARISON:
{'-'*50}
"""
    
    # Sort strategies by Sharpe ratio
    sorted_strategies = sorted(results.items(), 
                             key=lambda x: x[1]['sharpe_ratio'], 
                             reverse=True)
    
    for strategy_name, result in sorted_strategies:
        report += f"""
🔹 {strategy_name.upper()}:
   ✓ Total Return: {result['total_return']:.2f}%
   ✓ Annualized Return: {result['annualized_return']:.2f}%
   ✓ Volatility: {result['volatility']:.2f}%
   ✓ Sharpe Ratio: {result['sharpe_ratio']:.3f}
   ✓ Max Drawdown: {result['max_drawdown']:.2f}%
   ✓ Number of Trades: {result['num_trades']}
   ✓ Win Rate: {result['win_rate']:.1f}%
   ✓ Avg Win: {result['avg_win']:.2f}%
   ✓ Avg Loss: {result['avg_loss']:.2f}%
   ✓ Final Portfolio Value: ₹{result['final_value']:,.0f}
"""
    
    # Best performing strategy
    best_strategy = sorted_strategies[0]
    report += f"""
{'='*50}
BEST PERFORMING STRATEGY: {best_strategy[0].upper()}
Risk-Adjusted Return (Sharpe): {best_strategy[1]['sharpe_ratio']:.3f}
{'='*50}

RECOMMENDATION:
Based on risk-adjusted returns, the {best_strategy[0]} strategy 
shows the best performance with a Sharpe ratio of {best_strategy[1]['sharpe_ratio']:.3f}.

NEXT STEPS:
1. Optimize parameters for the best strategy
2. Implement proper risk management rules
3. Consider combining multiple indicators
4. Test on out-of-sample data
5. Implement position sizing rules

{'='*80}
"""
    
    return report

def main():
    """
    Main function demonstrating technical indicators implementation.
    """
    print("🔧 Technical Indicators Analysis - Assignment 2 Solution")
    print("="*70)
    
    # Generate sample data for demonstration
    print("📊 Generating sample market data...")
    dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
    np.random.seed(42)
    
    # Create realistic OHLCV data with trending behavior
    n_days = len(dates)
    returns = np.random.normal(0.001, 0.02, n_days)  # Slight upward bias
    
    # Add trending behavior
    trend = np.linspace(0, 0.5, n_days)  # Gradual uptrend
    returns += trend / n_days
    
    prices = [1000]  # Starting price
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # Generate OHLCV data
    data = pd.DataFrame({
        'Open': [p * (1 + np.random.normal(0, 0.003)) for p in prices],
        'High': [p * (1 + abs(np.random.normal(0.005, 0.008))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0.005, 0.008))) for p in prices],
        'Close': prices,
        'Volume': np.random.lognormal(15, 0.3, n_days).astype(int)
    }, index=dates)
    
    # Ensure OHLC logic
    data['High'] = np.maximum.reduce([data['Open'], data['High'], data['Close']])
    data['Low'] = np.minimum.reduce([data['Open'], data['Low'], data['Close']])
    
    print(f"✓ Sample data created: {len(data)} trading days")
    print(f"✓ Price range: ₹{data['Close'].min():.2f} - ₹{data['Close'].max():.2f}")
    
    # Step 1: Calculate Technical Indicators
    print("\n📈 Step 1: Calculating Technical Indicators")
    print("-" * 50)
    
    indicators = TechnicalIndicators(data)
    
    # RSI
    rsi = indicators.rsi(period=14)
    print(f"✓ RSI calculated - Current: {rsi.iloc[-1]:.2f}")
    
    # MACD
    macd_line, signal_line, histogram = indicators.macd()
    print(f"✓ MACD calculated - Current: {macd_line.iloc[-1]:.4f}")
    
    # Bollinger Bands
    bb_upper, bb_middle, bb_lower = indicators.bollinger_bands()
    current_bb_pos = indicators.data['BB_Percent_B'].iloc[-1]
    print(f"✓ Bollinger Bands calculated - Position: {current_bb_pos:.1f}%")
    
    # Stochastic
    stoch_k, stoch_d = indicators.stochastic_oscillator()
    print(f"✓ Stochastic calculated - %K: {stoch_k.iloc[-1]:.2f}")
    
    # ATR
    atr = indicators.atr()
    print(f"✓ ATR calculated - Current: {atr.iloc[-1]:.2f}")
    
    # Williams %R
    williams_r = indicators.williams_r()
    print(f"✓ Williams %R calculated - Current: {williams_r.iloc[-1]:.2f}")
    
    # Momentum
    momentum = indicators.momentum()
    print(f"✓ Momentum calculated - Current: {momentum.iloc[-1]:.2f}%")
    
    # Custom Trend Strength
    trend_strength = indicators.custom_trend_strength()
    print(f"✓ Custom Trend Strength - Current: {trend_strength.iloc[-1]:.1f}/100")
    
    # Step 2: Generate Trading Signals
    print("\n🎯 Step 2: Generating Trading Signals")
    print("-" * 50)
    
    signal_gen = SignalGenerator(indicators.data)
    
    # RSI signals
    rsi_signals = signal_gen.rsi_signals()
    rsi_trades = (rsi_signals != 0).sum()
    print(f"✓ RSI signals generated - {rsi_trades} trading signals")
    
    # MACD signals
    macd_signals = signal_gen.macd_signals()
    macd_trades = (macd_signals != 0).sum()
    print(f"✓ MACD signals generated - {macd_trades} trading signals")
    
    # Bollinger Bands signals
    bb_signals = signal_gen.bollinger_signals()
    bb_trades = (bb_signals != 0).sum()
    print(f"✓ Bollinger Bands signals generated - {bb_trades} trading signals")
    
    # Composite signals
    composite_signals = signal_gen.composite_signals()
    composite_trades = (composite_signals != 0).sum()
    print(f"✓ Composite signals generated - {composite_trades} trading signals")
    
    # Step 3: Backtest Strategies
    print("\n📊 Step 3: Backtesting Strategies")
    print("-" * 50)
    
    # Combine data with signals
    backtest_data = indicators.data.copy()
    backtest_data['RSI_Signal'] = signal_gen.signals['RSI_Signal']
    backtest_data['MACD_Signal'] = signal_gen.signals['MACD_Signal']
    backtest_data['BB_Signal'] = signal_gen.signals['BB_Signal']
    backtest_data['Composite_Signal'] = signal_gen.signals['Composite_Signal']
    
    backtester = IndicatorBacktester(backtest_data)
    
    # Backtest all strategies
    strategies = ['RSI_Signal', 'MACD_Signal', 'BB_Signal', 'Composite_Signal']
    all_results = {}
    
    for strategy in strategies:
        print(f"📈 Backtesting {strategy}...")
        result = backtester.backtest_strategy(strategy)
        all_results[strategy] = result
        print(f"   ✓ Total Return: {result['total_return']:.2f}%")
        print(f"   ✓ Sharpe Ratio: {result['sharpe_ratio']:.3f}")
        print(f"   ✓ Max Drawdown: {result['max_drawdown']:.2f}%")
    
    # Step 4: Generate Report
    print("\n📄 Step 4: Generating Analysis Report")
    print("-" * 50)
    
    report = generate_technical_report(all_results)
    print(report)
    
    # Step 5: Create Visualizations
    print("\n📊 Step 5: Creating Technical Dashboard")
    print("-" * 50)
    
    # Create dashboard
    dashboard = create_technical_dashboard(indicators.data, "SAMPLE_STOCK")
    print("✓ Interactive dashboard created")
    
    # Create performance comparison chart
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Portfolio values comparison
    plt.subplot(2, 2, 1)
    for strategy, result in all_results.items():
        portfolio_values = result['portfolio_values']
        returns = (portfolio_values / 100000 - 1) * 100
        plt.plot(returns.index, returns, label=strategy, linewidth=2)
    
    plt.title('Strategy Performance Comparison')
    plt.ylabel('Returns (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Technical indicators summary
    plt.subplot(2, 2, 2)
    current_values = {
        'RSI': indicators.data['RSI_14'].iloc[-1],
        'Stoch %K': indicators.data['Stoch_K'].iloc[-1],
        'Williams %R': abs(indicators.data['Williams_R_14'].iloc[-1]),
        'Trend Strength': indicators.data['Trend_Strength'].iloc[-1]
    }
    
    bars = plt.bar(current_values.keys(), current_values.values(), 
                   color=['red' if v > 70 else 'green' if v < 30 else 'blue' 
                         for v in current_values.values()])
    plt.title('Current Indicator Levels')
    plt.ylabel('Value')
    plt.ylim(0, 100)
    
    # Add reference lines
    plt.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought')
    plt.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Oversold')
    plt.legend()
    
    # Plot 3: Win Rate Comparison
    plt.subplot(2, 2, 3)
    win_rates = [result['win_rate'] for result in all_results.values()]
    strategy_names = [name.replace('_Signal', '') for name in all_results.keys()]
    
    bars = plt.bar(strategy_names, win_rates, 
                   color=['green' if wr > 50 else 'red' for wr in win_rates])
    plt.title('Win Rate by Strategy')
    plt.ylabel('Win Rate (%)')
    plt.axhline(y=50, color='black', linestyle='--', alpha=0.7)
    
    # Add value labels on bars
    for bar, wr in zip(bars, win_rates):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{wr:.1f}%', ha='center', va='bottom')
    
    # Plot 4: Risk-Return Scatter
    plt.subplot(2, 2, 4)
    returns = [result['annualized_return'] for result in all_results.values()]
    volatilities = [result['volatility'] for result in all_results.values()]
    
    plt.scatter(volatilities, returns, s=100, alpha=0.7)
    
    # Add strategy labels
    for i, strategy in enumerate(strategy_names):
        plt.annotate(strategy, (volatilities[i], returns[i]), 
                    xytext=(5, 5), textcoords='offset points')
    
    plt.title('Risk-Return Profile')
    plt.xlabel('Volatility (%)')
    plt.ylabel('Annualized Return (%)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("✅ Technical indicators analysis completed successfully!")
    print("\nKey Insights:")
    print(f"• Best performing strategy: {max(all_results.keys(), key=lambda k: all_results[k]['sharpe_ratio'])}")
    print(f"• Current market condition: {'Overbought' if indicators.data['RSI_14'].iloc[-1] > 70 else 'Oversold' if indicators.data['RSI_14'].iloc[-1] < 30 else 'Neutral'}")
    print(f"• Trend strength: {indicators.data['Trend_Strength'].iloc[-1]:.0f}/100")
    
    return indicators.data, all_results

if __name__ == "__main__":
    data, results = main()
