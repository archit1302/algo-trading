# Historical Data Downloader

This directory contains tools for downloading historical market data from Upstox API using the symbol mapper.

## Files

- `symbol_mapper.py`: Maps trading symbols to instrument keys
- `config.py`: General configuration file for data download parameters
- `config_sbin_15min.py`: Example configuration for SBIN 15-minute data
- `download_historical_data.py`: Main downloader class implementation
- `run_download.py`: Simple script to run the downloader with a specific config
- `batch_download.py`: Script to download data for multiple symbols

## Usage

### Basic Usage

1. Edit the `config.py` file to set your desired parameters:
   - SYMBOL: Trading symbol to download data for (e.g., "SBIN")
   - UNIT: Time unit (e.g., "minutes", "hours", "days", "weeks", "months")
   - INTERVAL: Interval number (e.g., "15" for 15-minute candles)
   - START_DATE and END_DATE: Date range in "YYYY-MM-DD" format
   - API_KEY: Your Upstox API key

2. Run the downloader:
   ```bash
   python3 run_download.py
   ```

3. Data will be saved to the `historical_data` directory in the specified format (CSV by default).

### Using Custom Config

You can create custom config files (like `config_sbin_15min.py`) and specify them when running the downloader:

```bash
python3 run_download.py config_sbin_15min
```

### Batch Download

To download data for multiple symbols at once:

```bash
python3 batch_download.py
```

Edit the symbols list in `batch_download.py` or use a text file with one symbol per line.

## API Limitations

Please be aware of the following limitations when downloading historical data:

| Unit    | Interval Options | Historical Availability | Max Retrieval Record Limit |
|---------|------------------|-------------------------|----------------------------|
| minutes | 1, 2, ... 300    | From January 2022       | 1 month (intervals 1-15)<br>1 quarter (intervals >15) |
| hours   | 1, 2, ... 5      | From January 2022       | 1 quarter |
| days    | 1                | From January 2000       | 1 decade |
| weeks   | 1                | From January 2000       | No limit |
| months  | 1                | From January 2000       | No limit |
