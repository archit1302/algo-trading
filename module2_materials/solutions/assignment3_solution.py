#!/usr/bin/env python3
"""
Assignment 3 Solution: Data Visualization for Financial Analysis
Module 2: Technical Analysis and Data Processing

This solution demonstrates professional financial data visualization using matplotlib,
plotly, and seaborn for creating interactive charts and dashboards.

Author: Financial Analytics Course
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class FinancialVisualizer:
    """
    Comprehensive financial data visualization toolkit.
    """
    
    def __init__(self, data, symbol="Stock"):
        """
        Initialize with financial data.
        
        Args:
            data (pd.DataFrame): OHLCV data with technical indicators
            symbol (str): Stock symbol for chart titles
        """
        self.data = data.copy()
        self.symbol = symbol
        
        # Ensure datetime index
        if not isinstance(self.data.index, pd.DatetimeIndex):
            self.data.index = pd.to_datetime(self.data.index)
    
    def candlestick_chart(self, figsize=(15, 8), volume=True, indicators=None):
        """
        Create professional candlestick chart with volume and indicators.
        
        Args:
            figsize (tuple): Figure size
            volume (bool): Include volume subplot
            indicators (list): List of indicator columns to plot
            
        Returns:
            matplotlib.figure.Figure: Chart figure
        """
        # Determine subplot layout
        n_subplots = 1 + (1 if volume else 0) + (len(indicators) if indicators else 0)
        height_ratios = [3] + [1] * (n_subplots - 1)
        
        fig, axes = plt.subplots(n_subplots, 1, figsize=figsize, 
                                height_ratios=height_ratios, 
                                sharex=True)
        
        if n_subplots == 1:
            axes = [axes]
        
        # Main candlestick chart
        ax_main = axes[0]
        
        # Prepare data for candlestick
        up_days = self.data['Close'] >= self.data['Open']
        down_days = self.data['Close'] < self.data['Open']
        
        # Plot candlestick bodies
        ax_main.bar(self.data.index[up_days], 
                   self.data['Close'][up_days] - self.data['Open'][up_days],
                   bottom=self.data['Open'][up_days],
                   color='green', alpha=0.8, width=0.8)
        
        ax_main.bar(self.data.index[down_days], 
                   self.data['Close'][down_days] - self.data['Open'][down_days],
                   bottom=self.data['Open'][down_days],
                   color='red', alpha=0.8, width=0.8)
        
        # Plot wicks
        ax_main.vlines(self.data.index, self.data['Low'], self.data['High'],
                      colors='black', linewidth=0.5)
        
        # Add moving averages if available
        ma_columns = [col for col in self.data.columns if 'SMA' in col or 'EMA' in col]
        colors = ['blue', 'orange', 'purple', 'brown']
        
        for i, ma_col in enumerate(ma_columns[:4]):  # Limit to 4 MAs
            if ma_col in self.data.columns:
                ax_main.plot(self.data.index, self.data[ma_col], 
                           label=ma_col, color=colors[i % len(colors)], 
                           linewidth=1.5, alpha=0.8)
        
        ax_main.set_title(f'{self.symbol} - Candlestick Chart', fontsize=16, fontweight='bold')
        ax_main.set_ylabel('Price (₹)', fontsize=12)
        ax_main.legend(loc='upper left')
        ax_main.grid(True, alpha=0.3)
        
        # Volume subplot
        subplot_idx = 1
        if volume and 'Volume' in self.data.columns:
            ax_vol = axes[subplot_idx]
            
            # Color volume bars based on price movement
            vol_colors = ['green' if close >= open_price else 'red' 
                         for close, open_price in zip(self.data['Close'], self.data['Open'])]
            
            ax_vol.bar(self.data.index, self.data['Volume'], 
                      color=vol_colors, alpha=0.7, width=0.8)
            
            ax_vol.set_ylabel('Volume', fontsize=12)
            ax_vol.set_title('Trading Volume', fontsize=12)
            ax_vol.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
            ax_vol.grid(True, alpha=0.3)
            
            subplot_idx += 1
        
        # Additional indicators
        if indicators:
            for indicator in indicators:
                if indicator in self.data.columns and subplot_idx < len(axes):
                    ax_ind = axes[subplot_idx]
                    ax_ind.plot(self.data.index, self.data[indicator], 
                              color='purple', linewidth=2)
                    ax_ind.set_ylabel(indicator, fontsize=12)
                    ax_ind.set_title(f'{indicator} Indicator', fontsize=12)
                    ax_ind.grid(True, alpha=0.3)
                    
                    # Add reference lines for common indicators
                    if 'RSI' in indicator:
                        ax_ind.axhline(y=70, color='red', linestyle='--', alpha=0.7)
                        ax_ind.axhline(y=30, color='green', linestyle='--', alpha=0.7)
                        ax_ind.set_ylim(0, 100)
                    
                    subplot_idx += 1
        
        # Format x-axis
        if len(axes) > 0:
            axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
            plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        return fig
    
    def interactive_dashboard(self):
        """
        Create interactive Plotly dashboard with multiple charts.
        
        Returns:
            plotly.graph_objects.Figure: Interactive dashboard
        """
        # Create subplots
        fig = make_subplots(
            rows=4, cols=2,
            subplot_titles=[
                'Price Chart with Technical Indicators',
                'Volume Analysis',
                'RSI Oscillator',
                'Price Distribution',
                'MACD Analysis',
                'Volatility Trends',
                'Support & Resistance Levels',
                'Performance Metrics'
            ],
            specs=[
                [{"colspan": 2}, None],
                [{"secondary_y": True}, {"type": "histogram"}],
                [{"secondary_y": True}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "indicator"}]
            ],
            vertical_spacing=0.08,
            row_heights=[0.4, 0.2, 0.2, 0.2]
        )
        
        # 1. Main price chart with candlesticks
        fig.add_trace(
            go.Candlestick(
                x=self.data.index,
                open=self.data['Open'],
                high=self.data['High'],
                low=self.data['Low'],
                close=self.data['Close'],
                name='Price',
                increasing_line_color='green',
                decreasing_line_color='red'
            ),
            row=1, col=1
        )
        
        # Add Bollinger Bands if available
        if 'BB_Upper_20' in self.data.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.data.index,
                    y=self.data['BB_Upper_20'],
                    name='BB Upper',
                    line=dict(color='red', width=1),
                    opacity=0.7
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=self.data.index,
                    y=self.data['BB_Lower_20'],
                    name='BB Lower',
                    line=dict(color='red', width=1),
                    fill='tonexty',
                    fillcolor='rgba(255,0,0,0.1)'
                ),
                row=1, col=1
            )
        
        # 2. Volume chart
        fig.add_trace(
            go.Bar(
                x=self.data.index,
                y=self.data['Volume'],
                name='Volume',
                marker_color='lightblue',
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # 3. RSI
        if 'RSI_14' in self.data.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.data.index,
                    y=self.data['RSI_14'],
                    name='RSI',
                    line=dict(color='purple')
                ),
                row=3, col=1
            )
            
            # RSI reference lines
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        
        # 4. Price distribution histogram
        if 'Daily_Return' in self.data.columns:
            returns = self.data['Daily_Return'].dropna() * 100
            fig.add_trace(
                go.Histogram(
                    x=returns,
                    name='Return Distribution',
                    nbinsx=50,
                    marker_color='green',
                    opacity=0.7
                ),
                row=2, col=2
            )
        
        # 5. MACD
        if 'MACD' in self.data.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.data.index,
                    y=self.data['MACD'],
                    name='MACD',
                    line=dict(color='blue')
                ),
                row=3, col=2
            )
            
            fig.add_trace(
                go.Scatter(
                    x=self.data.index,
                    y=self.data['MACD_Signal'],
                    name='Signal',
                    line=dict(color='red')
                ),
                row=3, col=2
            )
        
        # 6. Volatility analysis
        if 'Volatility_Annualized' in self.data.columns:
            vol_data = self.data['Volatility_Annualized'].dropna() * 100
            fig.add_trace(
                go.Scatter(
                    x=vol_data.index,
                    y=vol_data,
                    name='Volatility',
                    line=dict(color='orange'),
                    fill='tonexty'
                ),
                row=4, col=1
            )
        
        # 7. Support and Resistance levels
        recent_data = self.data.tail(60)  # Last 60 days
        support_level = recent_data['Low'].min()
        resistance_level = recent_data['High'].max()
        
        fig.add_hline(y=support_level, line_dash="dot", line_color="green", 
                     annotation_text="Support", row=1, col=1)
        fig.add_hline(y=resistance_level, line_dash="dot", line_color="red", 
                     annotation_text="Resistance", row=1, col=1)
        
        # 8. Performance indicator
        total_return = (self.data['Close'].iloc[-1] / self.data['Close'].iloc[0] - 1) * 100
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=total_return,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Total Return (%)"},
                delta={'reference': 0},
                gauge={
                    'axis': {'range': [-50, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [-50, 0], 'color': "lightgray"},
                        {'range': [0, 50], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 20
                    }
                }
            ),
            row=4, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=f'{self.symbol} - Comprehensive Financial Dashboard',
            xaxis_rangeslider_visible=False,
            height=1200,
            showlegend=True,
            template='plotly_white'
        )
        
        return fig
    
    def correlation_heatmap(self, indicators=None):
        """
        Create correlation heatmap of price and technical indicators.
        
        Args:
            indicators (list): List of columns to include in correlation
            
        Returns:
            matplotlib.figure.Figure: Heatmap figure
        """
        if indicators is None:
            # Select numeric columns excluding OHLCV
            indicators = [col for col in self.data.columns 
                         if col not in ['Open', 'High', 'Low', 'Close', 'Volume'] 
                         and self.data[col].dtype in ['float64', 'int64']]
        
        # Include Close price for correlation
        corr_data = self.data[['Close'] + indicators].corr()
        
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_data, dtype=bool))
        
        heatmap = sns.heatmap(corr_data, 
                             mask=mask,
                             annot=True, 
                             cmap='RdYlBu_r', 
                             center=0,
                             square=True,
                             linewidths=0.5,
                             cbar_kws={"shrink": 0.8})
        
        plt.title(f'{self.symbol} - Technical Indicators Correlation Matrix', 
                 fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return plt.gcf()
    
    def price_pattern_detection(self):
        """
        Visualize common price patterns and chart formations.
        
        Returns:
            matplotlib.figure.Figure: Pattern analysis chart
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Support and Resistance Levels
        ax1 = axes[0, 0]
        ax1.plot(self.data.index, self.data['Close'], linewidth=2, color='blue')
        
        # Calculate dynamic support/resistance
        window = 20
        rolling_max = self.data['High'].rolling(window=window).max()
        rolling_min = self.data['Low'].rolling(window=window).min()
        
        ax1.plot(self.data.index, rolling_max, 'r--', alpha=0.7, label='Resistance')
        ax1.plot(self.data.index, rolling_min, 'g--', alpha=0.7, label='Support')
        
        ax1.set_title('Dynamic Support & Resistance Levels')
        ax1.set_ylabel('Price (₹)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Price Channels
        ax2 = axes[0, 1]
        ax2.plot(self.data.index, self.data['Close'], linewidth=2, color='blue')
        
        # Linear regression channel
        if len(self.data) > 50:
            x_numeric = np.arange(len(self.data))
            coeffs = np.polyfit(x_numeric, self.data['Close'], 1)
            trend_line = np.poly1d(coeffs)
            
            # Calculate standard deviation for channel width
            residuals = self.data['Close'] - trend_line(x_numeric)
            std_dev = np.std(residuals)
            
            upper_channel = trend_line(x_numeric) + 2 * std_dev
            lower_channel = trend_line(x_numeric) - 2 * std_dev
            
            ax2.plot(self.data.index, trend_line(x_numeric), 'purple', 
                    linewidth=2, label='Trend Line')
            ax2.plot(self.data.index, upper_channel, 'r--', alpha=0.7, label='Upper Channel')
            ax2.plot(self.data.index, lower_channel, 'g--', alpha=0.7, label='Lower Channel')
        
        ax2.set_title('Price Channel Analysis')
        ax2.set_ylabel('Price (₹)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Breakout Detection
        ax3 = axes[1, 0]
        ax3.plot(self.data.index, self.data['Close'], linewidth=2, color='blue')
        
        # Identify breakouts (price moving beyond recent range)
        lookback = 20
        breakout_threshold = 0.02  # 2% breakout
        
        for i in range(lookback, len(self.data)):
            recent_high = self.data['High'].iloc[i-lookback:i].max()
            recent_low = self.data['Low'].iloc[i-lookback:i].min()
            current_price = self.data['Close'].iloc[i]
            
            # Upward breakout
            if current_price > recent_high * (1 + breakout_threshold):
                ax3.scatter(self.data.index[i], current_price, 
                           color='green', s=50, marker='^', alpha=0.7)
            
            # Downward breakout
            elif current_price < recent_low * (1 - breakout_threshold):
                ax3.scatter(self.data.index[i], current_price, 
                           color='red', s=50, marker='v', alpha=0.7)
        
        ax3.set_title('Breakout Detection')
        ax3.set_ylabel('Price (₹)')
        ax3.grid(True, alpha=0.3)
        
        # 4. Volume-Price Analysis
        ax4 = axes[1, 1]
        
        # Create volume-weighted price chart
        if 'Volume' in self.data.columns:
            # Calculate VWAP
            vwap = (self.data['Close'] * self.data['Volume']).cumsum() / self.data['Volume'].cumsum()
            
            ax4.plot(self.data.index, self.data['Close'], 'blue', linewidth=2, label='Price')
            ax4.plot(self.data.index, vwap, 'orange', linewidth=2, label='VWAP')
            
            # Highlight high volume days
            high_volume_threshold = self.data['Volume'].quantile(0.8)
            high_vol_days = self.data[self.data['Volume'] > high_volume_threshold]
            
            ax4.scatter(high_vol_days.index, high_vol_days['Close'], 
                       color='red', s=30, alpha=0.7, label='High Volume')
        
        ax4.set_title('Volume-Weighted Price Analysis')
        ax4.set_ylabel('Price (₹)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(f'{self.symbol} - Price Pattern Analysis', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def performance_dashboard(self):
        """
        Create comprehensive performance analysis dashboard.
        
        Returns:
            matplotlib.figure.Figure: Performance dashboard
        """
        fig = plt.figure(figsize=(20, 12))
        
        # Create grid layout
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. Cumulative Returns
        ax1 = fig.add_subplot(gs[0, :2])
        if 'Daily_Return' in self.data.columns:
            cumulative_returns = (1 + self.data['Daily_Return']).cumprod()
            ax1.plot(self.data.index, cumulative_returns, linewidth=2, color='blue')
            ax1.fill_between(self.data.index, 1, cumulative_returns, alpha=0.3)
            ax1.set_title('Cumulative Returns')
            ax1.set_ylabel('Cumulative Return')
            ax1.grid(True, alpha=0.3)
            ax1.axhline(y=1, color='black', linestyle='--', alpha=0.5)
        
        # 2. Rolling Volatility
        ax2 = fig.add_subplot(gs[0, 2:])
        if 'Volatility_Annualized' in self.data.columns:
            vol_data = self.data['Volatility_Annualized'] * 100
            ax2.plot(self.data.index, vol_data, color='red', linewidth=2)
            ax2.fill_between(self.data.index, 0, vol_data, alpha=0.3, color='red')
            ax2.set_title('Rolling Volatility (Annualized %)')
            ax2.set_ylabel('Volatility (%)')
            ax2.grid(True, alpha=0.3)
        
        # 3. Monthly Returns Heatmap
        ax3 = fig.add_subplot(gs[1, :2])
        if 'Daily_Return' in self.data.columns:
            monthly_returns = self.data['Daily_Return'].resample('M').apply(
                lambda x: (1 + x).prod() - 1
            ) * 100
            
            # Create month-year matrix for heatmap
            monthly_data = monthly_returns.to_frame()
            monthly_data['Year'] = monthly_data.index.year
            monthly_data['Month'] = monthly_data.index.month
            
            if len(monthly_data) > 0:
                pivot_table = monthly_data.pivot_table(
                    values='Daily_Return', 
                    index='Year', 
                    columns='Month', 
                    fill_value=0
                )
                
                sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='RdYlGn', 
                           center=0, ax=ax3, cbar_kws={'label': 'Monthly Return (%)'})
                ax3.set_title('Monthly Returns Heatmap')
        
        # 4. Drawdown Analysis
        ax4 = fig.add_subplot(gs[1, 2:])
        if 'Daily_Return' in self.data.columns:
            cumulative_returns = (1 + self.data['Daily_Return']).cumprod()
            peak = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - peak) / peak * 100
            
            ax4.fill_between(self.data.index, 0, drawdown, color='red', alpha=0.3)
            ax4.plot(self.data.index, drawdown, color='red', linewidth=1)
            ax4.set_title('Drawdown Analysis')
            ax4.set_ylabel('Drawdown (%)')
            ax4.grid(True, alpha=0.3)
        
        # 5. Return Distribution
        ax5 = fig.add_subplot(gs[2, 0])
        if 'Daily_Return' in self.data.columns:
            returns = self.data['Daily_Return'].dropna() * 100
            ax5.hist(returns, bins=50, alpha=0.7, color='blue', edgecolor='black')
            ax5.axvline(returns.mean(), color='red', linestyle='--', 
                       label=f'Mean: {returns.mean():.2f}%')
            ax5.set_title('Return Distribution')
            ax5.set_xlabel('Daily Return (%)')
            ax5.set_ylabel('Frequency')
            ax5.legend()
        
        # 6. Risk Metrics
        ax6 = fig.add_subplot(gs[2, 1])
        if 'Daily_Return' in self.data.columns:
            returns = self.data['Daily_Return'].dropna()
            
            metrics = {
                'Sharpe\nRatio': (returns.mean() * 252) / (returns.std() * np.sqrt(252)),
                'Sortino\nRatio': (returns.mean() * 252) / (returns[returns < 0].std() * np.sqrt(252)),
                'Skewness': returns.skew(),
                'Kurtosis': returns.kurtosis()
            }
            
            bars = ax6.bar(metrics.keys(), metrics.values(), 
                          color=['green' if v > 0 else 'red' for v in metrics.values()])
            ax6.set_title('Risk Metrics')
            ax6.set_ylabel('Value')
            
            # Add value labels on bars
            for bar, value in zip(bars, metrics.values()):
                ax6.text(bar.get_x() + bar.get_width()/2, 
                        bar.get_height() + 0.01 if value > 0 else bar.get_height() - 0.05,
                        f'{value:.2f}', ha='center', va='bottom' if value > 0 else 'top')
        
        # 7. Price Levels
        ax7 = fig.add_subplot(gs[2, 2])
        price_levels = {
            'Current': self.data['Close'].iloc[-1],
            '52W High': self.data['High'].max(),
            '52W Low': self.data['Low'].min(),
            '200D MA': self.data['Close'].rolling(200).mean().iloc[-1] if len(self.data) > 200 else np.nan
        }
        
        colors = ['blue', 'green', 'red', 'orange']
        bars = ax7.bar(range(len(price_levels)), list(price_levels.values()), 
                      color=colors)
        ax7.set_xticks(range(len(price_levels)))
        ax7.set_xticklabels(list(price_levels.keys()), rotation=45)
        ax7.set_title('Key Price Levels')
        ax7.set_ylabel('Price (₹)')
        
        # Add value labels
        for bar, value in zip(bars, price_levels.values()):
            if not np.isnan(value):
                ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                        f'₹{value:.2f}', ha='center', va='bottom')
        
        # 8. Technical Indicator Summary
        ax8 = fig.add_subplot(gs[2, 3])
        
        # Get current indicator values
        indicators = {}
        if 'RSI_14' in self.data.columns:
            indicators['RSI'] = self.data['RSI_14'].iloc[-1]
        if 'Stoch_K' in self.data.columns:
            indicators['Stoch %K'] = self.data['Stoch_K'].iloc[-1]
        if 'BB_Percent_B' in self.data.columns:
            indicators['BB %B'] = self.data['BB_Percent_B'].iloc[-1]
        
        if indicators:
            bars = ax8.barh(list(indicators.keys()), list(indicators.values()))
            ax8.set_title('Current Indicators')
            ax8.set_xlabel('Value')
            ax8.set_xlim(0, 100)
            
            # Add reference lines
            ax8.axvline(x=70, color='red', linestyle='--', alpha=0.7, label='Overbought')
            ax8.axvline(x=30, color='green', linestyle='--', alpha=0.7, label='Oversold')
            ax8.legend()
        
        plt.suptitle(f'{self.symbol} - Comprehensive Performance Dashboard', 
                     fontsize=20, fontweight='bold')
        
        return fig

def create_multi_asset_comparison(data_dict):
    """
    Create comparison charts for multiple assets.
    
    Args:
        data_dict (dict): Dictionary with asset names as keys and data as values
        
    Returns:
        plotly.graph_objects.Figure: Comparison dashboard
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Normalized Price Comparison',
            'Volatility Comparison',
            'Correlation Matrix',
            'Performance Metrics'
        ],
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"type": "heatmap"}, {"type": "bar"}]
        ]
    )
    
    # Prepare data for comparison
    normalized_prices = {}
    volatilities = {}
    returns_data = {}
    
    for asset_name, data in data_dict.items():
        # Normalize prices to start at 100
        normalized_prices[asset_name] = (data['Close'] / data['Close'].iloc[0]) * 100
        
        # Calculate rolling volatility
        if 'Daily_Return' in data.columns:
            returns_data[asset_name] = data['Daily_Return']
            volatilities[asset_name] = data['Daily_Return'].rolling(30).std() * np.sqrt(252) * 100
    
    # 1. Normalized price comparison
    colors = px.colors.qualitative.Set1
    for i, (asset_name, prices) in enumerate(normalized_prices.items()):
        fig.add_trace(
            go.Scatter(
                x=prices.index,
                y=prices,
                name=asset_name,
                line=dict(color=colors[i % len(colors)])
            ),
            row=1, col=1
        )
    
    # 2. Volatility comparison
    for i, (asset_name, vol) in enumerate(volatilities.items()):
        fig.add_trace(
            go.Scatter(
                x=vol.index,
                y=vol,
                name=f'{asset_name} Vol',
                line=dict(color=colors[i % len(colors)], dash='dash')
            ),
            row=1, col=2
        )
    
    # 3. Correlation matrix
    if len(returns_data) > 1:
        corr_df = pd.DataFrame(returns_data).corr()
        
        fig.add_trace(
            go.Heatmap(
                z=corr_df.values,
                x=corr_df.columns,
                y=corr_df.index,
                colorscale='RdYlBu_r',
                zmid=0,
                text=np.round(corr_df.values, 2),
                texttemplate="%{text}",
                textfont={"size": 12}
            ),
            row=2, col=1
        )
    
    # 4. Performance metrics comparison
    performance_metrics = {}
    for asset_name, data in data_dict.items():
        if 'Daily_Return' in data.columns:
            returns = data['Daily_Return'].dropna()
            total_return = (data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100
            volatility = returns.std() * np.sqrt(252) * 100
            sharpe = (returns.mean() * 252) / (returns.std() * np.sqrt(252))
            
            performance_metrics[asset_name] = {
                'Total Return (%)': total_return,
                'Volatility (%)': volatility,
                'Sharpe Ratio': sharpe
            }
    
    if performance_metrics:
        metrics_df = pd.DataFrame(performance_metrics).T
        
        for i, metric in enumerate(['Total Return (%)', 'Volatility (%)', 'Sharpe Ratio']):
            fig.add_trace(
                go.Bar(
                    x=list(performance_metrics.keys()),
                    y=metrics_df[metric],
                    name=metric,
                    offsetgroup=i
                ),
                row=2, col=2
            )
    
    fig.update_layout(
        title='Multi-Asset Comparison Dashboard',
        height=800,
        showlegend=True
    )
    
    return fig

def main():
    """
    Main function demonstrating financial visualization capabilities.
    """
    print("📊 Financial Data Visualization - Assignment 3 Solution")
    print("="*70)
    
    # Generate comprehensive sample data
    print("📈 Generating enhanced sample data...")
    
    dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
    np.random.seed(42)
    
    # Create realistic market data with trends and volatility clustering
    n_days = len(dates)
    base_vol = 0.02
    vol_persistence = 0.95
    
    # Generate volatility clustering
    volatilities = [base_vol]
    for _ in range(n_days - 1):
        new_vol = vol_persistence * volatilities[-1] + (1 - vol_persistence) * base_vol + \
                  0.001 * np.random.normal()
        volatilities.append(max(0.005, new_vol))  # Minimum volatility
    
    # Generate returns with volatility clustering
    returns = [np.random.normal(0.0008, vol) for vol in volatilities]
    
    # Add market trends
    trend_changes = np.random.choice(range(50, n_days, 100), size=5)
    for change_point in trend_changes:
        trend_strength = np.random.choice([-0.002, 0.002], p=[0.3, 0.7])
        for i in range(change_point, min(change_point + 50, n_days)):
            returns[i] += trend_strength
    
    # Generate prices
    prices = [1000]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # Create OHLCV data
    data = pd.DataFrame({
        'Open': [p * (1 + np.random.normal(0, 0.003)) for p in prices],
        'High': [p * (1 + abs(np.random.normal(0.005, 0.008))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0.005, 0.008))) for p in prices],
        'Close': prices,
        'Volume': np.random.lognormal(15, 0.4, n_days).astype(int)
    }, index=dates)
    
    # Ensure OHLC relationships
    data['High'] = np.maximum.reduce([data['Open'], data['High'], data['Close']])
    data['Low'] = np.minimum.reduce([data['Open'], data['Low'], data['Close']])
    
    # Add technical indicators for visualization
    data['Daily_Return'] = data['Close'].pct_change()
    data['SMA_20'] = data['Close'].rolling(20).mean()
    data['SMA_50'] = data['Close'].rolling(50).mean()
    data['EMA_12'] = data['Close'].ewm(span=12).mean()
    
    # RSI
    delta = data['Close'].diff()
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)
    avg_gains = gains.ewm(alpha=1/14).mean()
    avg_losses = losses.ewm(alpha=1/14).mean()
    rs = avg_gains / avg_losses
    data['RSI_14'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    data['BB_Middle_20'] = data['SMA_20']
    bb_std = data['Close'].rolling(20).std()
    data['BB_Upper_20'] = data['BB_Middle_20'] + (bb_std * 2)
    data['BB_Lower_20'] = data['BB_Middle_20'] - (bb_std * 2)
    data['BB_Percent_B'] = ((data['Close'] - data['BB_Lower_20']) / 
                           (data['BB_Upper_20'] - data['BB_Lower_20'])) * 100
    
    # MACD
    ema_fast = data['Close'].ewm(span=12).mean()
    ema_slow = data['Close'].ewm(span=26).mean()
    data['MACD'] = ema_fast - ema_slow
    data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
    data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
    
    # Stochastic
    lowest_low = data['Low'].rolling(14).min()
    highest_high = data['High'].rolling(14).max()
    data['Stoch_K'] = ((data['Close'] - lowest_low) / (highest_high - lowest_low)) * 100
    data['Stoch_D'] = data['Stoch_K'].rolling(3).mean()
    
    # Volatility
    data['Volatility_Rolling'] = data['Daily_Return'].rolling(20).std()
    data['Volatility_Annualized'] = data['Volatility_Rolling'] * np.sqrt(252)
    
    print(f"✓ Enhanced sample data created: {len(data)} trading days")
    print(f"✓ Technical indicators calculated")
    
    # Initialize visualizer
    visualizer = FinancialVisualizer(data, "SAMPLE_STOCK")
    
    # Step 1: Create Candlestick Chart
    print("\n📊 Step 1: Creating Professional Candlestick Chart")
    print("-" * 50)
    
    candlestick_fig = visualizer.candlestick_chart(
        volume=True, 
        indicators=['RSI_14']
    )
    plt.show()
    print("✓ Candlestick chart with volume and RSI created")
    
    # Step 2: Interactive Dashboard
    print("\n🎯 Step 2: Creating Interactive Dashboard")
    print("-" * 50)
    
    interactive_fig = visualizer.interactive_dashboard()
    interactive_fig.show()
    print("✓ Interactive Plotly dashboard created")
    
    # Step 3: Correlation Analysis
    print("\n📈 Step 3: Technical Indicators Correlation Analysis")
    print("-" * 50)
    
    indicators_list = ['RSI_14', 'Stoch_K', 'BB_Percent_B', 'MACD', 'Volatility_Annualized']
    correlation_fig = visualizer.correlation_heatmap(indicators_list)
    plt.show()
    print("✓ Correlation heatmap created")
    
    # Step 4: Price Pattern Analysis
    print("\n🔍 Step 4: Price Pattern Detection")
    print("-" * 50)
    
    pattern_fig = visualizer.price_pattern_detection()
    plt.show()
    print("✓ Price pattern analysis completed")
    
    # Step 5: Performance Dashboard
    print("\n📊 Step 5: Comprehensive Performance Dashboard")
    print("-" * 50)
    
    performance_fig = visualizer.performance_dashboard()
    plt.show()
    print("✓ Performance dashboard created")
    
    # Step 6: Multi-Asset Comparison (simulate multiple assets)
    print("\n🏢 Step 6: Multi-Asset Comparison")
    print("-" * 50)
    
    # Create synthetic data for comparison
    assets_data = {'SAMPLE_STOCK': data}
    
    # Generate data for comparison assets
    for asset_name in ['TECH_STOCK', 'BANK_STOCK']:
        # Modify the base data to create different assets
        asset_data = data.copy()
        noise_factor = np.random.normal(1, 0.1, len(data))
        asset_data['Close'] = data['Close'] * noise_factor
        asset_data['Daily_Return'] = asset_data['Close'].pct_change()
        assets_data[asset_name] = asset_data
    
    comparison_fig = create_multi_asset_comparison(assets_data)
    comparison_fig.show()
    print("✓ Multi-asset comparison dashboard created")
    
    # Generate summary statistics
    print("\n📊 Visualization Summary Statistics")
    print("-" * 50)
    
    total_return = (data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100
    current_rsi = data['RSI_14'].iloc[-1]
    current_bb_position = data['BB_Percent_B'].iloc[-1]
    avg_volume = data['Volume'].mean()
    volatility = data['Volatility_Annualized'].iloc[-1] * 100
    
    print(f"📈 Total Return: {total_return:.2f}%")
    print(f"📊 Current RSI: {current_rsi:.1f}")
    print(f"🎯 BB Position: {current_bb_position:.1f}%")
    print(f"📊 Average Volume: {avg_volume:,.0f}")
    print(f"📉 Current Volatility: {volatility:.1f}%")
    
    # Market condition assessment
    if current_rsi > 70:
        market_condition = "Overbought"
    elif current_rsi < 30:
        market_condition = "Oversold"
    else:
        market_condition = "Neutral"
    
    print(f"🎯 Market Condition: {market_condition}")
    
    # Chart recommendations
    print(f"\n💡 Chart Analysis Insights:")
    print(f"• Price vs SMA(20): {((data['Close'].iloc[-1] / data['SMA_20'].iloc[-1] - 1) * 100):+.1f}%")
    print(f"• Trend Direction: {'Bullish' if data['Close'].iloc[-1] > data['SMA_50'].iloc[-1] else 'Bearish'}")
    print(f"• MACD Signal: {'Bullish' if data['MACD'].iloc[-1] > data['MACD_Signal'].iloc[-1] else 'Bearish'}")
    
    print(f"\n✅ Financial visualization analysis completed successfully!")
    print(f"📊 Created {6} different chart types with professional styling")
    
    return data, visualizer

if __name__ == "__main__":
    data, visualizer = main()
