import os
import pandas as pd
from datetime import datetime

def read_and_print_files(directory_path):
    """
    Reads files from the directory and writes lines with close price > 815 to a separate file.
    
    Args:
        directory_path (str): Path to the directory containing data files
    """
    # Check if directory exists
    if not os.path.exists(directory_path):
        print(f"Directory '{directory_path}' not found!")
        return
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(directory_path), "high_price_data")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory for filtered data: {output_dir}")
    
    # List all files in the directory
    try:
        files = os.listdir(directory_path)
    except Exception as e:
        print(f"Error accessing directory: {e}")
        return
    
    # Filter for common data file types
    data_files = [f for f in files if os.path.isfile(os.path.join(directory_path, f))]
    
    if not data_files:
        print(f"No files found in {directory_path}")
        return
    
    print(f"Found {len(data_files)} files in {directory_path}\n")
    
    # Process each file
    for file in data_files:
        file_path = os.path.join(directory_path, file)
        print(f"\n{'='*50}")
        print(f"File: {file}")
        print(f"{'='*50}")
        
        try:
            # Handle different file types
            file_extension = os.path.splitext(file)[1].lower()
            
            if file_extension == '.csv':
                # Create output file for high price data
                high_price_filename = f"high_price_{file}"
                high_price_filepath = os.path.join(output_dir, high_price_filename)
                
                # Read with pandas to show structured data
                df = pd.read_csv(file_path)
                print(f"\nFile preview (first 5 rows):\n")
                print(df.head().to_string())
                
                # Check if we have a column that might be close price (4th column in typical OHLCV data)
                # Calculate number of filtered rows if close price column exists
                filtered_rows = 0
                with open(high_price_filepath, 'w') as out_file:
                    with open(file_path, 'r') as f:
                        # For each line in the file
                        for i, line in enumerate(f):
                            # Print first 10 lines to console
                            if i < 10:  
                                print(f"Line {i+1}: {line.strip()}")
                            
                            # Check if this is price data (skip header row)
                            parts = line.strip().split(',')
                            if len(parts) >= 5 and i > 0:  # Assuming at least 5 columns for OHLCV format
                                try:
                                    close_price = float(parts[4])  # Assuming 5th column is close price
                                    if close_price > 815:
                                        out_file.write(line)
                                        filtered_rows += 1
                                except (ValueError, IndexError):
                                    # Skip lines that don't have valid price data
                                    pass
                
                if i >= 10:
                    print(f"... and {i-9} more lines")
                
                if filtered_rows > 0:
                    print(f"\nFound {filtered_rows} rows with close price > 815")
                    print(f"Filtered data written to: {high_price_filepath}")
                else:
                    print("\nNo rows found with close price > 815")
                        
            elif file_extension in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
                print(f"\nFile preview (first 5 rows):\n")
                print(df.head().to_string())
                
                # Handle Excel files with high price filtering
                if len(df.columns) >= 5:  # Assuming at least 5 columns for OHLCV format
                    high_price_df = df[df.iloc[:, 4] > 815]  # Filter on 5th column (close price)
                    if not high_price_df.empty:
                        high_price_filename = f"high_price_{os.path.splitext(file)[0]}.csv"
                        high_price_filepath = os.path.join(output_dir, high_price_filename)
                        high_price_df.to_csv(high_price_filepath, index=False)
                        print(f"\nFound {len(high_price_df)} rows with close price > 815")
                        print(f"Filtered data written to: {high_price_filepath}")
                    else:
                        print("\nNo rows found with close price > 815")
                
            elif file_extension == '.txt':
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    for i, line in enumerate(lines[:10]):  # First 10 lines
                        print(f"Line {i+1}: {line.strip()}")
                    if len(lines) > 10:
                        print(f"... and {len(lines)-10} more lines")
            
            else:
                # For other file types, try to read as text
                try:
                    with open(file_path, 'r') as f:
                        lines = f.readlines()
                        for i, line in enumerate(lines[:10]):  # First 10 lines
                            print(f"Line {i+1}: {line.strip()}")
                        if len(lines) > 10:
                            print(f"... and {len(lines)-10} more lines")
                except UnicodeDecodeError:
                    print(f"Cannot display contents of {file} - not a text file or uses unsupported encoding")
        
        except Exception as e:
            print(f"Error reading file {file}: {str(e)}")

# Specify the correct path to the data_files directory
directory_path = "/Users/architmittal/Downloads/sample/module2/data_files"

# Call the function
read_and_print_files(directory_path)

# Assignments
# 1. Read the files from the data_files folder, and add a new file with a column which do average of open, high, low, close prices
# and write to a new file with avg_prices as column nameca