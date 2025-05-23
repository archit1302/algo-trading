#!/usr/bin/env python3
"""
Module 2 - Assignment 4 Solution: Time Series Analysis and Forecasting
Advanced time series analysis for financial data with forecasting capabilities.

Topics Covered:
- Time series decomposition
- Trend analysis and seasonal patterns
- Moving averages and exponential smoothing
- ARIMA modeling
- Volatility forecasting
- Prophet forecasting
- Model evaluation and validation

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
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_AVAILABLE = True
except ImportError:
    print("Warning: statsmodels not available. Some advanced features will be limited.")
    STATSMODELS_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    print("Warning: Prophet not available. Prophet forecasting will be limited.")
    PROPHET_AVAILABLE = False

class TimeSeriesAnalyzer:
    """
    Comprehensive time series analysis toolkit for financial data.
    """
    
    def __init__(self, data):
        """
        Initialize with financial time series data.
        
        Parameters:
        data (DataFrame): Financial data with datetime index and price columns
        """
        self.data = data.copy()
        self.ensure_datetime_index()
        
    def ensure_datetime_index(self):
        """Ensure the DataFrame has a proper datetime index."""
        if not isinstance(self.data.index, pd.DatetimeIndex):
            if 'date' in self.data.columns:
                self.data['date'] = pd.to_datetime(self.data['date'])
                self.data.set_index('date', inplace=True)
            elif 'Date' in self.data.columns:
                self.data['Date'] = pd.to_datetime(self.data['Date'])
                self.data.set_index('Date', inplace=True)
            else:
                # Assume first column is date
                first_col = self.data.columns[0]
                self.data[first_col] = pd.to_datetime(self.data[first_col])
                self.data.set_index(first_col, inplace=True)
        
        # Sort by date
        self.data.sort_index(inplace=True)
    
    def basic_stats(self):
        """Calculate basic time series statistics."""
        stats = {}
        
        for column in self.data.select_dtypes(include=[np.number]).columns:
            col_stats = {
                'mean': self.data[column].mean(),
                'std': self.data[column].std(),
                'min': self.data[column].min(),
                'max': self.data[column].max(),
                'skewness': self.data[column].skew(),
                'kurtosis': self.data[column].kurtosis()
            }
            
            # Calculate returns if it's a price column
            if any(price_word in column.lower() for price_word in ['close', 'price', 'open', 'high', 'low']):
                returns = self.data[column].pct_change().dropna()
                col_stats.update({
                    'daily_return_mean': returns.mean(),
                    'daily_return_std': returns.std(),
                    'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0,
                    'max_drawdown': self.calculate_max_drawdown(self.data[column])
                })
            
            stats[column] = col_stats
        
        return stats
    
    def calculate_max_drawdown(self, prices):
        """Calculate maximum drawdown for a price series."""
        cumulative = (1 + prices.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def decompose_series(self, column='close', model='additive', period=252):
        """
        Perform time series decomposition.
        
        Parameters:
        column (str): Column to decompose
        model (str): 'additive' or 'multiplicative'
        period (int): Seasonal period (252 for daily data = 1 year)
        """
        if not STATSMODELS_AVAILABLE:
            print("Statsmodels not available. Using simple trend calculation.")
            return self.simple_trend_analysis(column)
        
        try:
            # Ensure we have enough data
            if len(self.data) < 2 * period:
                period = min(len(self.data) // 2, 30)  # Use monthly seasonality as fallback
            
            decomposition = seasonal_decompose(
                self.data[column].dropna(), 
                model=model, 
                period=period,
                extrapolate_trend='freq'
            )
            
            # Plot decomposition
            fig, axes = plt.subplots(4, 1, figsize=(15, 12))
            
            decomposition.observed.plot(ax=axes[0], title='Original Series')
            decomposition.trend.plot(ax=axes[1], title='Trend')
            decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
            decomposition.resid.plot(ax=axes[3], title='Residual')
            
            plt.tight_layout()
            plt.show()
            
            return {
                'original': decomposition.observed,
                'trend': decomposition.trend,
                'seasonal': decomposition.seasonal,
                'residual': decomposition.resid
            }
            
        except Exception as e:
            print(f"Decomposition failed: {e}")
            return self.simple_trend_analysis(column)
    
    def simple_trend_analysis(self, column):
        """Simple trend analysis without statsmodels."""
        data = self.data[column].dropna()
        
        # Simple moving averages for trend
        trend = data.rolling(window=30, center=True).mean()
        
        # Seasonal component (simplified)
        seasonal = data.groupby(data.index.dayofyear).transform('mean') - data.mean()
        
        # Residual
        residual = data - trend - seasonal
        
        # Plot
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
        
        data.plot(ax=axes[0], title='Original Series')
        trend.plot(ax=axes[1], title='Trend (30-day MA)')
        seasonal.plot(ax=axes[2], title='Seasonal Pattern')
        residual.plot(ax=axes[3], title='Residual')
        
        plt.tight_layout()
        plt.show()
        
        return {
            'original': data,
            'trend': trend,
            'seasonal': seasonal,
            'residual': residual
        }
    
    def stationarity_test(self, column='close'):
        """Test for stationarity using Augmented Dickey-Fuller test."""
        if not STATSMODELS_AVAILABLE:
            print("Statsmodels not available. Cannot perform ADF test.")
            return None
        
        try:
            series = self.data[column].dropna()
            result = adfuller(series)
            
            print(f'Augmented Dickey-Fuller Test for {column}:')
            print(f'ADF Statistic: {result[0]:.6f}')
            print(f'p-value: {result[1]:.6f}')
            print(f'Critical Values:')
            for key, value in result[4].items():
                print(f'\t{key}: {value:.3f}')
            
            is_stationary = result[1] <= 0.05
            print(f'\nSeries is {"stationary" if is_stationary else "non-stationary"} at 5% significance level')
            
            return {
                'adf_statistic': result[0],
                'p_value': result[1],
                'critical_values': result[4],
                'is_stationary': is_stationary
            }
            
        except Exception as e:
            print(f"Stationarity test failed: {e}")
            return None
    
    def moving_averages_forecast(self, column='close', windows=[5, 10, 20, 50], forecast_days=30):
        """
        Simple moving averages forecasting.
        
        Parameters:
        column (str): Column to forecast
        windows (list): Moving average windows
        forecast_days (int): Number of days to forecast
        """
        data = self.data[column].dropna()
        
        # Calculate moving averages
        ma_data = pd.DataFrame(index=data.index)
        ma_data['actual'] = data
        
        for window in windows:
            if len(data) >= window:
                ma_data[f'MA_{window}'] = data.rolling(window=window).mean()
        
        # Generate forecast dates
        last_date = data.index[-1]
        forecast_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=forecast_days,
            freq='D'
        )
        
        # Simple forecast using last available MA values
        forecasts = {}
        for window in windows:
            if f'MA_{window}' in ma_data.columns:
                last_ma = ma_data[f'MA_{window}'].iloc[-1]
                if not pd.isna(last_ma):
                    forecasts[f'MA_{window}_forecast'] = [last_ma] * forecast_days
        
        # Plot historical and forecasts
        plt.figure(figsize=(15, 8))
        
        # Plot historical data
        plt.plot(data.index[-100:], data.iloc[-100:], label='Actual Price', linewidth=2)
        
        # Plot moving averages
        for window in windows:
            if f'MA_{window}' in ma_data.columns:
                plt.plot(data.index[-100:], ma_data[f'MA_{window}'].iloc[-100:], 
                        label=f'{window}-day MA', alpha=0.7)
        
        # Plot forecasts
        for forecast_name, forecast_values in forecasts.items():
            plt.plot(forecast_dates, forecast_values, 
                    label=forecast_name, linestyle='--', alpha=0.8)
        
        plt.title(f'{column.title()} - Moving Averages Forecast')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        return forecasts
    
    def exponential_smoothing_forecast(self, column='close', forecast_days=30):
        """
        Exponential smoothing forecast.
        
        Parameters:
        column (str): Column to forecast
        forecast_days (int): Number of days to forecast
        """
        if not STATSMODELS_AVAILABLE:
            print("Statsmodels not available. Using simple exponential smoothing.")
            return self.simple_exponential_smoothing(column, forecast_days)
        
        try:
            data = self.data[column].dropna()
            
            # Fit exponential smoothing model
            model = ExponentialSmoothing(
                data,
                trend='add',
                seasonal='add',
                seasonal_periods=min(30, len(data)//4)  # Monthly seasonality
            )
            fitted_model = model.fit()
            
            # Generate forecast
            forecast = fitted_model.forecast(steps=forecast_days)
            forecast_index = pd.date_range(
                start=data.index[-1] + timedelta(days=1),
                periods=forecast_days,
                freq='D'
            )
            
            # Plot results
            plt.figure(figsize=(15, 8))
            
            # Plot historical data
            plt.plot(data.index[-100:], data.iloc[-100:], label='Actual', linewidth=2)
            
            # Plot fitted values
            fitted_values = fitted_model.fittedvalues
            plt.plot(data.index[-100:], fitted_values.iloc[-100:], 
                    label='Fitted', alpha=0.7)
            
            # Plot forecast
            plt.plot(forecast_index, forecast, 
                    label=f'{forecast_days}-day Forecast', 
                    linestyle='--', linewidth=2)
            
            plt.title(f'{column.title()} - Exponential Smoothing Forecast')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
            
            return {
                'forecast': forecast,
                'forecast_index': forecast_index,
                'model_summary': fitted_model.summary()
            }
            
        except Exception as e:
            print(f"Exponential smoothing failed: {e}")
            return self.simple_exponential_smoothing(column, forecast_days)
    
    def simple_exponential_smoothing(self, column, forecast_days, alpha=0.3):
        """Simple exponential smoothing implementation."""
        data = self.data[column].dropna()
        
        # Calculate exponential smoothing
        smoothed = [data.iloc[0]]
        for i in range(1, len(data)):
            smoothed.append(alpha * data.iloc[i] + (1 - alpha) * smoothed[-1])
        
        # Forecast
        last_smoothed = smoothed[-1]
        forecast = [last_smoothed] * forecast_days
        
        forecast_index = pd.date_range(
            start=data.index[-1] + timedelta(days=1),
            periods=forecast_days,
            freq='D'
        )
        
        # Plot
        plt.figure(figsize=(15, 8))
        plt.plot(data.index[-100:], data.iloc[-100:], label='Actual', linewidth=2)
        plt.plot(data.index[-100:], smoothed[-100:], label='Smoothed', alpha=0.7)
        plt.plot(forecast_index, forecast, 
                label=f'{forecast_days}-day Forecast', 
                linestyle='--', linewidth=2)
        
        plt.title(f'{column.title()} - Simple Exponential Smoothing (α={alpha})')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        return {
            'forecast': forecast,
            'forecast_index': forecast_index,
            'smoothed_values': smoothed
        }
    
    def arima_forecast(self, column='close', order=(1,1,1), forecast_days=30):
        """
        ARIMA model forecasting.
        
        Parameters:
        column (str): Column to forecast
        order (tuple): ARIMA order (p,d,q)
        forecast_days (int): Number of days to forecast
        """
        if not STATSMODELS_AVAILABLE:
            print("Statsmodels not available. Cannot perform ARIMA forecasting.")
            return None
        
        try:
            data = self.data[column].dropna()
            
            # Fit ARIMA model
            model = ARIMA(data, order=order)
            fitted_model = model.fit()
            
            # Generate forecast
            forecast_result = fitted_model.forecast(steps=forecast_days)
            forecast_index = pd.date_range(
                start=data.index[-1] + timedelta(days=1),
                periods=forecast_days,
                freq='D'
            )
            
            # Get confidence intervals
            forecast_ci = fitted_model.get_forecast(steps=forecast_days).conf_int()
            
            # Plot results
            plt.figure(figsize=(15, 8))
            
            # Plot historical data
            plt.plot(data.index[-100:], data.iloc[-100:], label='Actual', linewidth=2)
            
            # Plot forecast
            plt.plot(forecast_index, forecast_result, 
                    label=f'ARIMA{order} Forecast', 
                    linestyle='--', linewidth=2)
            
            # Plot confidence intervals
            plt.fill_between(forecast_index, 
                           forecast_ci.iloc[:, 0], 
                           forecast_ci.iloc[:, 1], 
                           alpha=0.3, label='95% Confidence Interval')
            
            plt.title(f'{column.title()} - ARIMA{order} Forecast')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
            
            print(f"\nARIMA{order} Model Summary:")
            print(fitted_model.summary())
            
            return {
                'forecast': forecast_result,
                'forecast_index': forecast_index,
                'confidence_intervals': forecast_ci,
                'model_summary': fitted_model.summary(),
                'aic': fitted_model.aic,
                'bic': fitted_model.bic
            }
            
        except Exception as e:
            print(f"ARIMA forecasting failed: {e}")
            return None
    
    def volatility_forecast(self, column='close', method='garch', forecast_days=30):
        """
        Volatility forecasting using various methods.
        
        Parameters:
        column (str): Column to analyze
        method (str): 'simple', 'ewma', or 'garch'
        forecast_days (int): Number of days to forecast
        """
        data = self.data[column].dropna()
        returns = data.pct_change().dropna()
        
        if method == 'simple':
            # Simple historical volatility
            vol_window = min(30, len(returns)//2)
            historical_vol = returns.rolling(window=vol_window).std()
            last_vol = historical_vol.iloc[-1]
            forecast_vol = [last_vol] * forecast_days
            
        elif method == 'ewma':
            # Exponentially Weighted Moving Average
            lambda_param = 0.94
            ewma_var = returns.ewm(alpha=1-lambda_param).var()
            ewma_vol = np.sqrt(ewma_var)
            last_vol = ewma_vol.iloc[-1]
            forecast_vol = [last_vol] * forecast_days
            
        else:
            print(f"Method {method} not implemented. Using EWMA.")
            lambda_param = 0.94
            ewma_var = returns.ewm(alpha=1-lambda_param).var()
            ewma_vol = np.sqrt(ewma_var)
            last_vol = ewma_vol.iloc[-1]
            forecast_vol = [last_vol] * forecast_days
        
        # Generate forecast dates
        forecast_index = pd.date_range(
            start=data.index[-1] + timedelta(days=1),
            periods=forecast_days,
            freq='D'
        )
        
        # Plot volatility
        plt.figure(figsize=(15, 8))
        
        # Plot historical volatility
        if method == 'simple':
            plt.plot(returns.index[-100:], historical_vol.iloc[-100:] * np.sqrt(252), 
                    label='Historical Volatility (30-day)', linewidth=2)
        elif method == 'ewma':
            plt.plot(returns.index[-100:], ewma_vol.iloc[-100:] * np.sqrt(252), 
                    label='EWMA Volatility', linewidth=2)
        
        # Plot forecast
        plt.plot(forecast_index, np.array(forecast_vol) * np.sqrt(252), 
                label=f'Volatility Forecast ({method.upper()})', 
                linestyle='--', linewidth=2)
        
        plt.title(f'{column.title()} - Volatility Forecast ({method.upper()})')
        plt.xlabel('Date')
        plt.ylabel('Annualized Volatility')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        return {
            'forecast_volatility': forecast_vol,
            'forecast_index': forecast_index,
            'annualized_forecast': np.array(forecast_vol) * np.sqrt(252)
        }
    
    def prophet_forecast(self, column='close', forecast_days=30):
        """
        Facebook Prophet forecasting.
        
        Parameters:
        column (str): Column to forecast
        forecast_days (int): Number of days to forecast
        """
        if not PROPHET_AVAILABLE:
            print("Prophet not available. Using simple trend forecasting.")
            return self.simple_trend_forecast(column, forecast_days)
        
        try:
            # Prepare data for Prophet
            data = self.data[column].dropna().reset_index()
            data.columns = ['ds', 'y']
            
            # Initialize and fit Prophet model
            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=True,
                changepoint_prior_scale=0.05
            )
            model.fit(data)
            
            # Create future dataframe
            future = model.make_future_dataframe(periods=forecast_days)
            
            # Generate forecast
            forecast = model.predict(future)
            
            # Plot results
            fig = model.plot(forecast, figsize=(15, 8))
            plt.title(f'{column.title()} - Prophet Forecast')
            plt.show()
            
            # Plot components
            fig2 = model.plot_components(forecast)
            plt.show()
            
            return {
                'forecast': forecast,
                'model': model,
                'future_dates': future
            }
            
        except Exception as e:
            print(f"Prophet forecasting failed: {e}")
            return self.simple_trend_forecast(column, forecast_days)
    
    def simple_trend_forecast(self, column, forecast_days):
        """Simple linear trend forecasting."""
        data = self.data[column].dropna()
        
        # Calculate linear trend
        x = np.arange(len(data))
        coeffs = np.polyfit(x, data.values, 1)
        trend = np.polyval(coeffs, x)
        
        # Forecast
        future_x = np.arange(len(data), len(data) + forecast_days)
        forecast = np.polyval(coeffs, future_x)
        
        forecast_index = pd.date_range(
            start=data.index[-1] + timedelta(days=1),
            periods=forecast_days,
            freq='D'
        )
        
        # Plot
        plt.figure(figsize=(15, 8))
        plt.plot(data.index[-100:], data.iloc[-100:], label='Actual', linewidth=2)
        plt.plot(data.index[-100:], trend[-100:], label='Trend', alpha=0.7)
        plt.plot(forecast_index, forecast, 
                label=f'{forecast_days}-day Trend Forecast', 
                linestyle='--', linewidth=2)
        
        plt.title(f'{column.title()} - Linear Trend Forecast')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        return {
            'forecast': forecast,
            'forecast_index': forecast_index,
            'trend_coefficients': coeffs
        }
    
    def model_evaluation(self, actual, predicted, model_name="Model"):
        """
        Evaluate forecasting model performance.
        
        Parameters:
        actual (array-like): Actual values
        predicted (array-like): Predicted values
        model_name (str): Name of the model
        """
        # Ensure same length
        min_len = min(len(actual), len(predicted))
        actual = actual[-min_len:]
        predicted = predicted[-min_len:]
        
        # Calculate metrics
        mae = np.mean(np.abs(actual - predicted))
        mse = np.mean((actual - predicted) ** 2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        
        # Directional accuracy
        actual_direction = np.diff(actual) > 0
        predicted_direction = np.diff(predicted) > 0
        directional_accuracy = np.mean(actual_direction == predicted_direction) * 100
        
        print(f"\n{model_name} Evaluation Metrics:")
        print(f"MAE (Mean Absolute Error): {mae:.4f}")
        print(f"MSE (Mean Squared Error): {mse:.4f}")
        print(f"RMSE (Root Mean Squared Error): {rmse:.4f}")
        print(f"MAPE (Mean Absolute Percentage Error): {mape:.2f}%")
        print(f"Directional Accuracy: {directional_accuracy:.2f}%")
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape,
            'directional_accuracy': directional_accuracy
        }
    
    def correlation_analysis(self):
        """Analyze correlations between different time series."""
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            print("Need at least 2 numeric columns for correlation analysis.")
            return None
        
        # Calculate correlation matrix
        correlation_matrix = self.data[numeric_cols].corr()
        
        # Plot correlation heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5)
        plt.title('Time Series Correlation Matrix')
        plt.tight_layout()
        plt.show()
        
        return correlation_matrix

def generate_sample_data():
    """Generate sample financial data for demonstration."""
    # Generate sample stock price data
    np.random.seed(42)
    dates = pd.date_range(start='2022-01-01', end='2024-12-31', freq='D')
    
    # Generate price with trend and seasonality
    trend = np.linspace(100, 150, len(dates))
    seasonal = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 252)  # Yearly seasonality
    noise = np.random.normal(0, 5, len(dates))
    
    price = trend + seasonal + noise
    
    # Generate OHLC data
    data = pd.DataFrame({
        'open': price + np.random.normal(0, 1, len(dates)),
        'high': price + np.abs(np.random.normal(2, 1, len(dates))),
        'low': price - np.abs(np.random.normal(2, 1, len(dates))),
        'close': price,
        'volume': np.random.randint(1000000, 5000000, len(dates))
    }, index=dates)
    
    # Ensure OHLC logic
    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)
    
    return data

def main():
    """Main function demonstrating time series analysis capabilities."""
    print("=== Module 2 Assignment 4: Time Series Analysis and Forecasting ===\n")
    
    # Generate or load sample data
    print("1. Loading sample financial data...")
    data = generate_sample_data()
    
    # Initialize analyzer
    analyzer = TimeSeriesAnalyzer(data)
    
    # Basic statistics
    print("\n2. Basic Time Series Statistics:")
    stats = analyzer.basic_stats()
    for column, column_stats in stats.items():
        print(f"\n{column.upper()}:")
        for stat_name, stat_value in column_stats.items():
            print(f"  {stat_name}: {stat_value:.4f}")
    
    # Time series decomposition
    print("\n3. Time Series Decomposition:")
    decomp_result = analyzer.decompose_series('close', period=30)
    
    # Stationarity test
    print("\n4. Stationarity Testing:")
    stationarity = analyzer.stationarity_test('close')
    
    # Moving averages forecast
    print("\n5. Moving Averages Forecasting:")
    ma_forecast = analyzer.moving_averages_forecast('close', windows=[5, 10, 20], forecast_days=30)
    
    # Exponential smoothing forecast
    print("\n6. Exponential Smoothing Forecasting:")
    es_forecast = analyzer.exponential_smoothing_forecast('close', forecast_days=30)
    
    # ARIMA forecast
    print("\n7. ARIMA Forecasting:")
    arima_forecast = analyzer.arima_forecast('close', order=(1,1,1), forecast_days=30)
    
    # Volatility forecast
    print("\n8. Volatility Forecasting:")
    vol_forecast = analyzer.volatility_forecast('close', method='ewma', forecast_days=30)
    
    # Prophet forecast
    print("\n9. Prophet Forecasting:")
    prophet_forecast = analyzer.prophet_forecast('close', forecast_days=30)
    
    # Correlation analysis
    print("\n10. Correlation Analysis:")
    correlation_matrix = analyzer.correlation_analysis()
    
    print("\n=== Time Series Analysis Complete ===")
    print("This analysis covers:")
    print("- Time series decomposition and trend analysis")
    print("- Stationarity testing")
    print("- Multiple forecasting methods (MA, ES, ARIMA, Prophet)")
    print("- Volatility forecasting")
    print("- Model evaluation and correlation analysis")

if __name__ == "__main__":
    main()
