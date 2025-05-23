
# load all the files from the data files folder
# create a proper list of all the value which are present 
# in the data at 3:00 PM IST 

import os 
import pandas as pd
from datetime import datetime

def extract_3pm_values(directory_path=None):
    """
    Extracts values from all data files in the directory that correspond to 3:00 PM IST.
    
    Args:
        directory_path (str): Path to the directory containing data files
        
    Returns:
        dict: Dictionary with filenames as keys and their 3:00 PM values as values
    """
    # Set the proper path for the data_files directory
    if directory_path is None:
        # Use the absolute path to the data_files directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        directory_path = os.path.join(current_dir, '..', 'data_files')  # Go up one level and to data_files
    
    print(f"Searching for data files in: {directory_path}")
    
    # Check if directory exists
    if not os.path.exists(directory_path):
        print(f"Directory '{directory_path}' not found!")
        # Try alternate path
        alternate_path = os.path.join(os.path.dirname(current_dir), 'data_files')
        print(f"Trying alternate path: {alternate_path}")
        
        if os.path.exists(alternate_path):
            directory_path = alternate_path
            print(f"Using alternate path: {directory_path}")
        else:
            return {}
    
    # Dictionary to store results
    values_at_3pm = {}
    
    # List all files in the directory
    files = os.listdir(directory_path)
    data_files = [f for f in files if f.endswith('.csv') or f.endswith('.xlsx')]
    
    if not data_files:
        print(f"No data files found in '{directory_path}'")
        return {}
    
    # Process each file
    for file in data_files:
        file_path = os.path.join(directory_path, file)
        try:
            # Determine file type and read accordingly
            if file.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:  # Excel file
                df = pd.read_excel(file_path)
            
            # Ensure there's a datetime column
            time_columns = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
            
            if not time_columns:
                print(f"No datetime column found in {file}")
                continue
                
            time_col = time_columns[0]
            
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
                df[time_col] = pd.to_datetime(df[time_col])
            
            # Filter for rows at 3:00 PM - fix the syntax error in the filter
            three_pm_data = df[(df[time_col].dt.hour == 15) & (df[time_col].dt.minute == 0)]
            
            if three_pm_data.empty:
                print(f"No data at 3:00 PM found in {file}")
                values_at_3pm[file] = None
            else:
                # Store all columns except the datetime column
                values_at_3pm[file] = three_pm_data.drop(columns=[time_col]).to_dict('records')
                
        except Exception as e:
            print(f"Error processing file {file}: {str(e)}")
            values_at_3pm[file] = f"Error: {str(e)}"
    
    return values_at_3pm

if __name__ == "__main__":
    # Try different possible paths
    possible_paths = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data_files'),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data_files'),
        '/Users/architmittal/Downloads/sample/data_files',
        '/Users/architmittal/Downloads/sample/module2/data_files'
    ]
    
    result = None
    
    for path in possible_paths:
        print(f"Trying path: {path}")
        if os.path.exists(path):
            print(f"Found directory: {path}")
            result = extract_3pm_values(path)
            if result:  # If we found data files, break the loop
                break
    
    if not result:
        print("Could not find data files in any of the expected locations.")
    else:
        # Print the results
        print("\nValues at 3:00 PM IST from all data files:")
        print("="*50)
        
        for file, values in result.items():
            print(f"\nFile: {file}")
            if values is None:
                print("  No data at 3:00 PM")
            elif isinstance(values, str) and values.startswith("Error"):
                print(f"  {values}")
            else:
                for i, record in enumerate(values):
                    print(f"  Record {i+1}: {record}")



# YYYY MM-DD HH:MM:SS -> 2025 04 17 15:00:00
# this will help you sort the files easily

# Assignments
# 1. Write a function to extract values from all data files in the directory that correspond to 2:00 PM IST.
# 2. Write a function to extract values from all data files in the directory with customizable time.