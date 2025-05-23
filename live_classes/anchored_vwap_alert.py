import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

def load_data(file_path):
    """Load the OHLC data from CSV file"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    df = pd.read_csv(file_path)
    
    # Ensure datetime column is properly formatted
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df

def calculate_anchored_vwap(df, anchor_date):
    """Calculate Anchored VWAP from a specific date/time point"""
    # Convert anchor_date to pandas datetime if it's a string
    if isinstance(anchor_date, str):
        anchor_date = pd.to_datetime(anchor_date)
        
    # Create a mask for data after the anchor point
    mask = df['datetime'] >= anchor_date
    
    # Initialize columns for calculations
    df['vwap_sum'] = np.nan
    df['volume_sum'] = np.nan
    df['anchored_vwap'] = np.nan
    
    # Calculate OHLC4 (average of Open, High, Low, Close)
    df['ohlc4'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    
    # Filter dataframe to get data points after anchor date
    if mask.any():
        df_after_anchor = df[mask].copy()
        
        # Calculate cumulative sum of (price * volume) and volume
        df_after_anchor['vwap_sum'] = (df_after_anchor['ohlc4'] * df_after_anchor['Volume']).cumsum()
        df_after_anchor['volume_sum'] = df_after_anchor['Volume'].cumsum()
        
        # Calculate VWAP
        df_after_anchor['anchored_vwap'] = df_after_anchor['vwap_sum'] / df_after_anchor['volume_sum']
        
        # Update the main dataframe
        df.loc[mask, 'vwap_sum'] = df_after_anchor['vwap_sum']
        df.loc[mask, 'volume_sum'] = df_after_anchor['volume_sum']
        df.loc[mask, 'anchored_vwap'] = df_after_anchor['anchored_vwap']
        
    return df

def detect_vwap_crossovers(df):
    """Detect when price crosses above or below the Anchored VWAP"""
    # Create shifted columns to detect crossovers
    df['Prev_Close'] = df['Close'].shift(1)
    df['Prev_AVWAP'] = df['anchored_vwap'].shift(1)
    
    # Detect crossover above (previous close was below VWAP, current close is above VWAP)
    df['Crossover_Above_VWAP'] = (df['Prev_Close'] < df['Prev_AVWAP']) & (df['Close'] > df['anchored_vwap'])
    
    # Detect crossover below (previous close was above VWAP, current close is below VWAP)
    df['Crossover_Below_VWAP'] = (df['Prev_Close'] > df['Prev_AVWAP']) & (df['Close'] < df['anchored_vwap'])
    
    # Calculate distance from VWAP (for signal strength)
    df['Price_VWAP_Diff'] = df['Close'] - df['anchored_vwap']
    
    return df

def plot_vwap_with_crossovers(df, title='Close Price vs Anchored VWAP with Crossover Points'):
    """Create a plot showing price, Anchored VWAP and crossover points"""
    # Filter out rows where AVWAP is NaN
    df_plot = df.dropna(subset=['anchored_vwap'])
    
    if len(df_plot) == 0:
        print("No VWAP data available for plotting. Check if anchor date is within the data range.")
        return None
    
    plt.figure(figsize=(12, 6))
    
    # Plot Close price and AVWAP
    plt.plot(df_plot['datetime'], df_plot['Close'], label='Close Price', color='blue')
    plt.plot(df_plot['datetime'], df_plot['anchored_vwap'], label='Anchored VWAP', color='purple', linewidth=2)
    
    # Highlight crossover points
    crossover_above = df_plot[df_plot['Crossover_Above_VWAP'] == True]
    crossover_below = df_plot[df_plot['Crossover_Below_VWAP'] == True]
    
    plt.scatter(crossover_above['datetime'], crossover_above['Close'], 
                color='green', marker='^', s=100, label='Crossover Above VWAP')
    
    plt.scatter(crossover_below['datetime'], crossover_below['Close'], 
                color='red', marker='v', s=100, label='Crossover Below VWAP')
    
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
    plot_path = f'anchored_vwap_plot_{timestamp}.png'
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Plot saved as: {plot_path}")
    return plot_path

def main(file_path='SBIN_20250415.csv', anchor_date='2025-04-15 09:30:00', show_plot=True):
    """Main function to load data, calculate Anchored VWAP and detect crossovers"""
    # Load data
    print(f"Loading data from {file_path}...")
    df = load_data(file_path)
    print(f"Loaded {len(df)} data points")
    
    # Check if the anchor date is in the data range
    anchor_datetime = pd.to_datetime(anchor_date)
    if anchor_datetime < df['datetime'].min() or anchor_datetime > df['datetime'].max():
        print(f"Warning: Anchor date {anchor_date} is outside the data range ({df['datetime'].min()} to {df['datetime'].max()})")
        
    # Calculate Anchored VWAP
    print(f"Calculating Anchored VWAP from {anchor_date}...")
    df = calculate_anchored_vwap(df, anchor_date)
    
    # Get the first valid VWAP value (after anchor point)
    first_valid = df.dropna(subset=['anchored_vwap'])['anchored_vwap'].iloc[0] if not df.dropna(subset=['anchored_vwap']).empty else None
    print(f"Initial Anchored VWAP value: {first_valid:.2f}" if first_valid is not None else "No valid VWAP values calculated")
    
    # Detect crossovers
    df = detect_vwap_crossovers(df)
    
    # Filter for crossover events
    crossover_above_events = df[df['Crossover_Above_VWAP'] == True].dropna(subset=['anchored_vwap'])
    crossover_below_events = df[df['Crossover_Below_VWAP'] == True].dropna(subset=['anchored_vwap'])
    
    # Print alert messages for crossover above VWAP
    if len(crossover_above_events) > 0:
        print(f"\n=== CLOSE PRICE CROSSED ABOVE ANCHORED VWAP ALERTS ===")
        for idx, row in crossover_above_events.iterrows():
            strength = row['Price_VWAP_Diff']
            strength_msg = "Strong" if abs(strength) > 5.0 else "Moderate" if abs(strength) > 2.0 else "Weak"
            print(f"ALERT at {row['datetime']}: Close price {row['Close']} crossed above Anchored VWAP {row['anchored_vwap']:.2f} " +
                 f"(Diff: {strength:.2f} - {strength_msg})")
        print(f"Total crossover above alerts: {len(crossover_above_events)}")
    else:
        print(f"No crossovers above Anchored VWAP detected.")
        
    # Print alert messages for crossover below VWAP
    if len(crossover_below_events) > 0:
        print(f"\n=== CLOSE PRICE CROSSED BELOW ANCHORED VWAP ALERTS ===")
        for idx, row in crossover_below_events.iterrows():
            strength = abs(row['Price_VWAP_Diff'])
            strength_msg = "Strong" if strength > 5.0 else "Moderate" if strength > 2.0 else "Weak"
            print(f"ALERT at {row['datetime']}: Close price {row['Close']} crossed below Anchored VWAP {row['anchored_vwap']:.2f} " +
                 f"(Diff: {strength:.2f} - {strength_msg})")
        print(f"Total crossover below alerts: {len(crossover_below_events)}")
    else:
        print(f"No crossovers below Anchored VWAP detected.")
    
    # Create plot if requested
    if show_plot and df['anchored_vwap'].notna().any():
        file_date = os.path.splitext(os.path.basename(file_path))[0]
        anchor_time = anchor_datetime.strftime("%H:%M")
        plot_path = plot_vwap_with_crossovers(df, f'{file_date} - Anchored VWAP (from {anchor_time}) with Crossover Points')
        
    # Return the dataframe for potential further analysis
    return df

if __name__ == "__main__":
    # Parameters based on the PineScript code
    file_path = 'SBIN_20250415.csv'
    
    # Default anchor date (corresponding to Year=2025, Month=4, Day=15, Hour=9, Minute=30)
    # You can modify this to match your specific requirements
    anchor_date = '2025-04-15 09:30:00'  # Format: YYYY-MM-DD HH:MM:SS
    
    show_plot = True
    
    df_result = main(file_path, anchor_date, show_plot)
