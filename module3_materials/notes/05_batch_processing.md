# Batch Processing and Data Management

This note covers advanced techniques for batch processing multiple symbols, managing large datasets, and implementing efficient data workflows with the Upstox API v3.

## Introduction to Batch Processing

Batch processing involves handling multiple symbols, timeframes, or data requests in an organized and efficient manner. This is essential for:

- Portfolio analysis across multiple stocks
- Market screening and scanning
- Large-scale backtesting
- Data pipeline automation
- Research and analytics

## Batch Processing Architecture

### Core Components

1. **Configuration Management**: Defining what data to fetch
2. **Task Queue**: Managing multiple requests
3. **Rate Limiting**: Respecting API limitations
4. **Error Handling**: Managing failures gracefully
5. **Data Storage**: Organizing and saving results
6. **Progress Tracking**: Monitoring batch progress

### Basic Batch Processor Structure

```python
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import logging

class BatchProcessor:
    """
    A comprehensive batch processor for Upstox historical data
    """
    
    def __init__(self, access_token, max_workers=3, request_delay=1.0):
        self.access_token = access_token
        self.max_workers = max_workers
        self.request_delay = request_delay
        self.results = {}
        self.errors = []
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def process_symbols(self, symbol_configs, parallel=True):
        """
        Process multiple symbols
        
        Args:
            symbol_configs (list): List of symbol configuration dictionaries
            parallel (bool): Whether to process in parallel
        
        Returns:
            dict: Results for each symbol
        """
        if parallel:
            return self._process_parallel(symbol_configs)
        else:
            return self._process_sequential(symbol_configs)
```

## Configuration Management

### Symbol Configuration Structure

```python
class SymbolConfig:
    """
    Configuration for a symbol's data requirements
    """
    
    def __init__(self, symbol, instrument_key, timeframes=None, 
                 date_range=None, custom_params=None):
        self.symbol = symbol
        self.instrument_key = instrument_key
        self.timeframes = timeframes or [{'interval': 1, 'unit': 'day'}]
        self.date_range = date_range or {}
        self.custom_params = custom_params or {}
    
    def to_dict(self):
        return {
            'symbol': self.symbol,
            'instrument_key': self.instrument_key,
            'timeframes': self.timeframes,
            'date_range': self.date_range,
            'custom_params': self.custom_params
        }

# Example configuration
configs = [
    SymbolConfig(
        symbol='SBIN',
        instrument_key='NSE_EQ|INE062A01020|SBIN',
        timeframes=[
            {'interval': 1, 'unit': 'day'},
            {'interval': 1, 'unit': 'hour'},
            {'interval': 5, 'unit': 'minute'}
        ],
        date_range={
            'from_date': '2024-01-01',
            'to_date': '2024-12-31'
        }
    ),
    SymbolConfig(
        symbol='RELIANCE',
        instrument_key='NSE_EQ|INE002A01018|RELIANCE',
        timeframes=[{'interval': 1, 'unit': 'day'}],
        date_range={'from_date': '2024-01-01'}
    )
]
```

### Configuration Files

```python
import json
import yaml

def load_config_from_file(filepath, format='json'):
    """
    Load batch configuration from file
    
    Args:
        filepath (str): Path to configuration file
        format (str): File format ('json' or 'yaml')
    
    Returns:
        list: List of symbol configurations
    """
    try:
        with open(filepath, 'r') as f:
            if format.lower() == 'json':
                data = json.load(f)
            elif format.lower() == 'yaml':
                data = yaml.safe_load(f)
            else:
                raise ValueError("Unsupported format. Use 'json' or 'yaml'")
        
        # Convert to SymbolConfig objects
        configs = []
        for item in data.get('symbols', []):
            config = SymbolConfig(
                symbol=item['symbol'],
                instrument_key=item['instrument_key'],
                timeframes=item.get('timeframes', []),
                date_range=item.get('date_range', {}),
                custom_params=item.get('custom_params', {})
            )
            configs.append(config)
        
        return configs
        
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return []

# Example JSON configuration file
config_json = {
    "symbols": [
        {
            "symbol": "SBIN",
            "instrument_key": "NSE_EQ|INE062A01020|SBIN",
            "timeframes": [
                {"interval": 1, "unit": "day"},
                {"interval": 1, "unit": "hour"}
            ],
            "date_range": {
                "from_date": "2024-01-01",
                "to_date": "2024-12-31"
            }
        }
    ]
}
```

## Parallel Processing Implementation

### Thread-Based Parallel Processing

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

class ParallelBatchProcessor:
    """
    Parallel batch processor using threading
    """
    
    def __init__(self, access_token, max_workers=5, request_delay=1.0):
        self.access_token = access_token
        self.max_workers = max_workers
        self.request_delay = request_delay
        self.lock = threading.Lock()
        self.results = {}
        self.errors = []
        
    def _fetch_single_symbol(self, config):
        """
        Fetch data for a single symbol configuration
        """
        symbol = config.symbol
        
        try:
            symbol_results = {}
            
            for timeframe in config.timeframes:
                timeframe_key = f"{timeframe['interval']}{timeframe['unit']}"
                
                # Add delay to respect rate limits
                time.sleep(self.request_delay)
                
                # Fetch data
                data = self._fetch_historical_data(
                    config.instrument_key,
                    timeframe['interval'],
                    timeframe['unit'],
                    config.date_range.get('from_date'),
                    config.date_range.get('to_date')
                )
                
                if data:
                    symbol_results[timeframe_key] = data
                else:
                    self.errors.append(f"Failed to fetch {symbol} {timeframe_key}")
            
            # Thread-safe result storage
            with self.lock:
                self.results[symbol] = symbol_results
                
            return symbol, symbol_results
            
        except Exception as e:
            error_msg = f"Error processing {symbol}: {e}"
            with self.lock:
                self.errors.append(error_msg)
            return symbol, None
    
    def process_batch(self, symbol_configs):
        """
        Process multiple symbols in parallel
        """
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_symbol = {
                executor.submit(self._fetch_single_symbol, config): config.symbol
                for config in symbol_configs
            }
            
            # Process completed tasks
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    symbol, result = future.result()
                    if result:
                        print(f"✓ Completed {symbol}")
                    else:
                        print(f"✗ Failed {symbol}")
                except Exception as e:
                    print(f"✗ Exception for {symbol}: {e}")
        
        return self.results, self.errors
```

### Progress Tracking

```python
from tqdm import tqdm
import time

class ProgressBatchProcessor:
    """
    Batch processor with progress tracking
    """
    
    def __init__(self, access_token, max_workers=3):
        self.access_token = access_token
        self.max_workers = max_workers
        self.progress_bar = None
    
    def process_with_progress(self, symbol_configs):
        """
        Process symbols with progress bar
        """
        results = {}
        errors = []
        
        # Calculate total operations
        total_operations = sum(len(config.timeframes) for config in symbol_configs)
        
        with tqdm(total=total_operations, desc="Fetching data") as pbar:
            for config in symbol_configs:
                symbol_results = {}
                
                for timeframe in config.timeframes:
                    try:
                        # Update progress description
                        timeframe_key = f"{timeframe['interval']}{timeframe['unit']}"
                        pbar.set_description(f"Fetching {config.symbol} {timeframe_key}")
                        
                        # Fetch data
                        data = self._fetch_historical_data(
                            config.instrument_key,
                            timeframe['interval'],
                            timeframe['unit'],
                            config.date_range.get('from_date'),
                            config.date_range.get('to_date')
                        )
                        
                        if data:
                            symbol_results[timeframe_key] = data
                        
                    except Exception as e:
                        errors.append(f"Error fetching {config.symbol} {timeframe_key}: {e}")
                    
                    finally:
                        pbar.update(1)
                        time.sleep(1)  # Rate limiting
                
                results[config.symbol] = symbol_results
        
        return results, errors
```

## Data Storage and Organization

### Structured Data Storage

```python
import os
import pandas as pd
from pathlib import Path

class DataManager:
    """
    Manages storage and organization of batch processed data
    """
    
    def __init__(self, base_dir="data"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # Create organized directory structure
        self.raw_dir = self.base_dir / "raw"
        self.processed_dir = self.base_dir / "processed"
        self.metadata_dir = self.base_dir / "metadata"
        
        for dir_path in [self.raw_dir, self.processed_dir, self.metadata_dir]:
            dir_path.mkdir(exist_ok=True)
    
    def save_batch_results(self, results, batch_id=None):
        """
        Save batch processing results with organized structure
        
        Args:
            results (dict): Results from batch processing
            batch_id (str): Optional batch identifier
        
        Returns:
            dict: Mapping of saved files
        """
        if batch_id is None:
            batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        saved_files = {}
        
        for symbol, timeframe_data in results.items():
            symbol_dir = self.raw_dir / symbol
            symbol_dir.mkdir(exist_ok=True)
            
            saved_files[symbol] = {}
            
            for timeframe, data in timeframe_data.items():
                if isinstance(data, pd.DataFrame) and not data.empty:
                    filename = f"{symbol}_{timeframe}_{batch_id}.csv"
                    filepath = symbol_dir / filename
                    
                    # Save data
                    data.to_csv(filepath)
                    saved_files[symbol][timeframe] = str(filepath)
                    
                    print(f"Saved {len(data)} records to {filepath}")
        
        # Save metadata
        metadata = {
            'batch_id': batch_id,
            'timestamp': datetime.now().isoformat(),
            'symbols_processed': list(results.keys()),
            'files_saved': saved_files
        }
        
        metadata_file = self.metadata_dir / f"batch_{batch_id}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return saved_files
    
    def load_symbol_data(self, symbol, timeframe=None):
        """
        Load saved data for a symbol
        
        Args:
            symbol (str): Symbol name
            timeframe (str): Optional timeframe filter
        
        Returns:
            dict or DataFrame: Loaded data
        """
        symbol_dir = self.raw_dir / symbol
        
        if not symbol_dir.exists():
            print(f"No data found for symbol: {symbol}")
            return None
        
        if timeframe:
            # Load specific timeframe
            pattern = f"{symbol}_{timeframe}_*.csv"
            files = list(symbol_dir.glob(pattern))
            
            if files:
                # Load the most recent file
                latest_file = max(files, key=os.path.getctime)
                return pd.read_csv(latest_file, index_col=0, parse_dates=True)
            else:
                print(f"No data found for {symbol} {timeframe}")
                return pd.DataFrame()
        else:
            # Load all timeframes
            data = {}
            for csv_file in symbol_dir.glob("*.csv"):
                # Extract timeframe from filename
                parts = csv_file.stem.split('_')
                if len(parts) >= 3:
                    tf = parts[1]
                    data[tf] = pd.read_csv(csv_file, index_col=0, parse_dates=True)
            
            return data
```

## Error Handling and Recovery

### Robust Error Management

```python
import logging
from datetime import datetime

class RobustBatchProcessor:
    """
    Batch processor with comprehensive error handling
    """
    
    def __init__(self, access_token, max_workers=3):
        self.access_token = access_token
        self.max_workers = max_workers
        self.setup_logging()
        
        # Error tracking
        self.retry_queue = []
        self.failed_requests = []
        self.success_count = 0
        self.error_count = 0
    
    def setup_logging(self):
        """Setup comprehensive logging"""
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(f'batch_processing_{datetime.now().strftime("%Y%m%d")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def process_with_retry(self, symbol_configs, max_retries=3):
        """
        Process symbols with automatic retry for failed requests
        """
        results = {}
        
        # Initial processing
        self.logger.info(f"Starting batch processing for {len(symbol_configs)} symbols")
        
        for config in symbol_configs:
            success = self._process_symbol_with_retry(config, max_retries)
            if success:
                self.success_count += 1
            else:
                self.error_count += 1
        
        # Summary logging
        self.logger.info(f"Batch processing completed. Success: {self.success_count}, Errors: {self.error_count}")
        
        return results
    
    def _process_symbol_with_retry(self, config, max_retries):
        """
        Process a single symbol with retry logic
        """
        symbol = config.symbol
        
        for attempt in range(max_retries + 1):
            try:
                self.logger.info(f"Processing {symbol} (attempt {attempt + 1})")
                
                # Process symbol
                result = self._fetch_symbol_data(config)
                
                if result:
                    self.logger.info(f"✓ Successfully processed {symbol}")
                    return True
                else:
                    self.logger.warning(f"No data returned for {symbol}")
                    
            except Exception as e:
                self.logger.error(f"Error processing {symbol} (attempt {attempt + 1}): {e}")
                
                if attempt < max_retries:
                    # Exponential backoff
                    wait_time = 2 ** attempt
                    self.logger.info(f"Retrying {symbol} in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    self.failed_requests.append({
                        'symbol': symbol,
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    })
                    return False
        
        return False
```

## Performance Optimization

### Memory Management

```python
import gc
import psutil

class OptimizedBatchProcessor:
    """
    Memory-optimized batch processor for large datasets
    """
    
    def __init__(self, access_token, memory_limit_mb=1024):
        self.access_token = access_token
        self.memory_limit_mb = memory_limit_mb
        self.current_memory_mb = 0
    
    def process_large_batch(self, symbol_configs, chunk_size=10):
        """
        Process large batches in chunks to manage memory
        """
        results = {}
        
        # Process in chunks
        for i in range(0, len(symbol_configs), chunk_size):
            chunk = symbol_configs[i:i + chunk_size]
            
            print(f"Processing chunk {i//chunk_size + 1}/{(len(symbol_configs)-1)//chunk_size + 1}")
            
            # Process chunk
            chunk_results = self._process_chunk(chunk)
            results.update(chunk_results)
            
            # Memory management
            self._check_memory_usage()
            
            # Force garbage collection
            gc.collect()
        
        return results
    
    def _check_memory_usage(self):
        """Check and manage memory usage"""
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        if memory_mb > self.memory_limit_mb:
            print(f"Warning: Memory usage ({memory_mb:.1f} MB) exceeds limit ({self.memory_limit_mb} MB)")
            gc.collect()
```

## Scheduling and Automation

### Automated Batch Jobs

```python
import schedule
import time
from datetime import datetime

class ScheduledBatchProcessor:
    """
    Scheduler for automated batch processing
    """
    
    def __init__(self, access_token, config_file):
        self.access_token = access_token
        self.config_file = config_file
        self.processor = BatchProcessor(access_token)
    
    def daily_update_job(self):
        """
        Daily job to update historical data
        """
        print(f"Starting daily update job at {datetime.now()}")
        
        try:
            # Load configuration
            configs = load_config_from_file(self.config_file)
            
            # Update date range to fetch only recent data
            for config in configs:
                config.date_range['from_date'] = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
                config.date_range['to_date'] = datetime.now().strftime("%Y-%m-%d")
            
            # Process batch
            results, errors = self.processor.process_batch(configs)
            
            print(f"Daily update completed. Processed {len(results)} symbols with {len(errors)} errors.")
            
        except Exception as e:
            print(f"Daily update job failed: {e}")
    
    def start_scheduler(self):
        """
        Start the batch processing scheduler
        """
        # Schedule daily updates
        schedule.every().day.at("18:00").do(self.daily_update_job)
        
        # Schedule weekly full updates
        schedule.every().sunday.at("20:00").do(self.weekly_full_update)
        
        print("Scheduler started. Press Ctrl+C to stop.")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            print("Scheduler stopped.")
```

## Best Practices

1. **Rate Limiting**: Always implement delays between requests
2. **Error Recovery**: Use retry logic with exponential backoff
3. **Memory Management**: Process large batches in chunks
4. **Logging**: Implement comprehensive logging for debugging
5. **Data Validation**: Validate data after each fetch
6. **Progress Tracking**: Provide feedback on long-running operations
7. **Configuration Management**: Use external config files for flexibility
8. **Resource Monitoring**: Monitor CPU and memory usage
9. **Parallel Processing**: Use threading for I/O-bound operations
10. **Data Organization**: Structure saved data for easy retrieval

## Key Takeaways

- Batch processing enables efficient handling of multiple symbols and timeframes
- Proper error handling and retry logic are essential for robust operations
- Memory management becomes critical when processing large datasets
- Progress tracking and logging improve user experience and debugging
- Structured data storage facilitates easy data retrieval and analysis
- Automation through scheduling reduces manual intervention
