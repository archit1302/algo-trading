import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime

def load_data(file_path):
    """Load the OHLC data from CSV file"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    df = pd.read_csv(file_path)
    
    # Ensure datetime column is properly formatted
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df

def calculate_indicators(df, ema_period=9):
    """Calculate technical indicators including EMA"""
    # Calculate EMA
    df['EMA_9'] = df['Close'].ewm(span=ema_period, adjust=False).mean()
    
    # Calculate price difference from EMA (can be used to measure strength of crossover)
    df['Price_EMA_Diff'] = df['Close'] - df['EMA_9']
    
    return df

def detect_crossover(df):
    """Detect when close price crosses above or below 9 EMA"""
    # Create shifted columns to detect crossovers
    df['Prev_Close'] = df['Close'].shift(1)
    df['Prev_EMA_9'] = df['EMA_9'].shift(1)
    
    # Detect crossover above (previous close was below EMA, current close is above EMA)
    df['Crossover_Above'] = (df['Prev_Close'] < df['Prev_EMA_9']) & (df['Close'] > df['EMA_9'])
    
    # Detect crossover below (previous close was above EMA, current close is below EMA)
    df['Crossover_Below'] = (df['Prev_Close'] > df['Prev_EMA_9']) & (df['Close'] < df['EMA_9'])
    
    return df

def plot_crossovers(df, title='Close Price vs 9 EMA with Crossover Points'):
    """Create a plot showing price, EMA and crossover points"""
    plt.figure(figsize=(12, 6))
    
    # Plot Close price and EMA
    plt.plot(df['datetime'], df['Close'], label='Close Price', color='blue')
    plt.plot(df['datetime'], df['EMA_9'], label='9 EMA', color='orange')
    
    # Highlight crossover points
    crossover_above = df[df['Crossover_Above'] == True]
    crossover_below = df[df['Crossover_Below'] == True]
    
    plt.scatter(crossover_above['datetime'], crossover_above['Close'], 
                color='green', marker='^', s=100, label='Crossover Above')
    
    plt.scatter(crossover_below['datetime'], crossover_below['Close'], 
                color='red', marker='v', s=100, label='Crossover Below')
    
    plt.title(title)
    plt.xlabel('DateTime')
    plt.ylabel('Price')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Format x-axis to show fewer timestamps
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f'ema_crossover_plot_{timestamp}.png'
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Plot saved as: {plot_path}")
    return plot_path

def main(file_path='SBIN_20250415.csv', ema_period=9, show_plot=True):
    """Main function to load data, calculate indicators and detect crossovers"""
    # Load data
    print(f"Loading data from {file_path}...")
    df = load_data(file_path)
    print(f"Loaded {len(df)} data points")
    
    # Calculate indicators
    df = calculate_indicators(df, ema_period)
    
    # Detect crossovers
    df = detect_crossover(df)
    
    # Filter for crossover events
    crossover_above_events = df[df['Crossover_Above'] == True]
    crossover_below_events = df[df['Crossover_Below'] == True]
    
    # Print alert messages for crossover above
    if len(crossover_above_events) > 0:
        print(f"\n=== CLOSE PRICE CROSSED ABOVE {ema_period} EMA ALERTS ===")
        for idx, row in crossover_above_events.iterrows():
            strength = row['Price_EMA_Diff']
            strength_msg = "Strong" if strength > 1.0 else "Moderate" if strength > 0.5 else "Weak"
            print(f"ALERT at {row['datetime']}: Close price {row['Close']} crossed above {ema_period} EMA {row['EMA_9']:.2f} " +
                 f"(Diff: {strength:.2f} - {strength_msg})")
        print(f"Total crossover above alerts: {len(crossover_above_events)}")
    else:
        print(f"No crossovers above {ema_period} EMA detected.")
        
    # Print alert messages for crossover below
    if len(crossover_below_events) > 0:
        print(f"\n=== CLOSE PRICE CROSSED BELOW {ema_period} EMA ALERTS ===")
        for idx, row in crossover_below_events.iterrows():
            strength = abs(row['Price_EMA_Diff'])
            strength_msg = "Strong" if strength > 1.0 else "Moderate" if strength > 0.5 else "Weak"
            print(f"ALERT at {row['datetime']}: Close price {row['Close']} crossed below {ema_period} EMA {row['EMA_9']:.2f} " +
                 f"(Diff: {strength:.2f} - {strength_msg})")
        print(f"Total crossover below alerts: {len(crossover_below_events)}")
    else:
        print(f"No crossovers below {ema_period} EMA detected.")
    
    # Create plot if requested
    if show_plot:
        file_date = os.path.splitext(os.path.basename(file_path))[0]
        plot_path = plot_crossovers(df, f'{file_date} - Close Price vs {ema_period} EMA with Crossover Points')
        
    # Return the dataframe for potential further analysis
    return df

if __name__ == "__main__":
    # You can customize these parameters
    file_path = 'SBIN_20250415.csv'
    ema_period = 9
    show_plot = True
    
    df_result = main(file_path, ema_period, show_plot)
