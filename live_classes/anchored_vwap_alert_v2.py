import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import argparse

def load_data(file_path):
    """Load the OHLC data from CSV file"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    df = pd.read_csv(file_path)
    
    # print(df.shape)
    # Ensure datetime column is properly formatted
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df

def calculate_anchored_vwap(df, year, month, day, hour, minute):
    """Calculate Anchored VWAP from a specific date/time point"""
    # Create anchor date from specified components
    try:
        anchor_date = pd.Timestamp(year=year, month=month, day=day, hour=hour, minute=minute)
        print(f"Using anchor date: {anchor_date}")
    except ValueError as e:
        print(f"Error creating timestamp: {e}")
        print("Using first timestamp in data as anchor point instead")
        anchor_date = df['datetime'].iloc[0]
        
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

def main(file_path='SBIN_20250415.csv', year=2025, month=4, day=15, hour=9, minute=30, show_plot=True, debug_mode=False):
    """Main function to load data, calculate Anchored VWAP and detect crossovers"""
    # Load data
    print(f"Loading data from {file_path}...")
    df = load_data(file_path)
    print(f"Loaded {len(df)} data points")
    
    # Calculate Anchored VWAP
    print(f"Calculating Anchored VWAP from {year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}...")
    df = calculate_anchored_vwap(df, year, month, day, hour, minute)
    
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
    
    # Debug mode output (similar to the PineScript debug mode)
    if debug_mode:
        print("\n=== DEBUG MODE OUTPUT ===")
        # Print the hour and minute values as in the PineScript debug output
        debug_df = df.dropna(subset=['anchored_vwap']).copy()
        debug_df['hour'] = debug_df['datetime'].dt.hour
        debug_df['minute'] = debug_df['datetime'].dt.minute
        
        print("Sample of hour values:", debug_df['hour'].iloc[:10].tolist())
        print("Sample of minute values:", debug_df['minute'].iloc[:10].tolist())
    
    # Create plot if requested
    if show_plot and df['anchored_vwap'].notna().any():
        file_date = os.path.splitext(os.path.basename(file_path))[0]
        anchor_time = f"{year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}"
        plot_path = plot_vwap_with_crossovers(df, f'{file_date} - Anchored VWAP (from {anchor_time}) with Crossover Points')
        
    # Return the dataframe for potential further analysis
    return df

if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description='Anchored VWAP Alert System')
    
    # Add arguments, matching the PineScript inputs
    parser.add_argument('--file', type=str, default='SBIN_20250415.csv', help='Path to CSV file with OHLC data')
    parser.add_argument('--year', type=int, default=2025, help='Anchor year')
    parser.add_argument('--month', type=int, default=4, help='Anchor month')
    parser.add_argument('--day', type=int, default=15, help='Anchor day')
    parser.add_argument('--hour', type=int, default=9, help='Anchor hour (24-hour format)')
    parser.add_argument('--minute', type=int, default=30, help='Anchor minute')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--no-plot', action='store_true', help='Disable plotting')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run main function with parsed arguments
    df_result = main(
        file_path=args.file, 
        year=args.year, 
        month=args.month, 
        day=args.day, 
        hour=args.hour, 
        minute=args.minute,
        show_plot=not args.no_plot,
        debug_mode=args.debug
    )
