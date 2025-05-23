import os

def setup_module1():
    """Setup script for Module 1: Python Fundamentals for Financial Data Processing"""
    
    print("Setting up Module 1: Python Fundamentals for Financial Data Processing")
    print("=" * 60)
    
    # Required packages for Module 1
    packages = [
        'pandas>=1.3.0',      # For data manipulation
        'matplotlib>=3.3.0',  # For basic plotting
        'numpy>=1.21.0',      # For numerical operations
    ]
    
    print("\nRequired packages:")
    for package in packages:
        print(f"  - {package}")
    
    print("\nTo install all dependencies, run:")
    print("pip install " + " ".join([p.split('>=')[0] for p in packages]))
    
    # Check if data directory exists
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"\nCreated {data_dir} directory for sample data files")
    
    print("\nSetup instructions:")
    print("1. Install the required packages using pip")
    print("2. Copy sample SBIN data from ../module1/data files/ to ./data/")
    print("3. Start with notes/01_python_basics.md")
    print("4. Complete assignments in order (assignment1 through assignment7)")
    print("5. Check your solutions against the provided solution files")
    
    print("\nSample data files to copy:")
    print("  - SBIN_20250415.csv (main sample file)")
    print("  - Other SBIN daily files for practice")
    
    print("\nReady to start learning Python for financial data processing!")

if __name__ == "__main__":
    setup_module1()
