# Module 3 Materials - README

This folder contains comprehensive materials for learning data fetching and processing for algorithmic trading using Python, with a focus on Upstox API v3.

## Structure

The folder is organized into three main directories:

### 1. Notes
Detailed explanation of key concepts with examples:

- `01_upstox_api_introduction.md` - Introduction to Upstox API v3
- `02_upstox_authentication.md` - Authentication process for Upstox API
- `03_instrument_mapping.md` - Understanding instrument mapping in Upstox
- `04_historical_data_fetching.md` - Fetching historical data using Upstox API v3
- `05_batch_processing.md` - Batch processing multiple symbols and timeframes
- `06_data_resampling.md` - Converting between different timeframes (5m to 1h, etc.)
- `07_market_scanner.md` - Creating market scanners with technical indicators

### 2. Assignments
Practice exercises with increasing difficulty:

- `assignment1_upstox_basics.md` - Setting up and using Upstox API v3
- `assignment2_upstox_authentication.md` - Implementing Upstox OAuth authentication
- `assignment3_instrument_mapping.md` - Working with Upstox instrument mapping 
- `assignment4_historical_data.md` - Fetching historical data for single symbol
- `assignment5_batch_processing.md` - Batch processing for multiple symbols
- `assignment6_data_resampling.md` - Timeframe conversion and resampling
- `assignment7_market_scanner.md` - Building a technical indicator scanner

### 3. Solutions
Complete implementations for all assignments:

- `assignment1_api_basics.py` - Solution for API basics assignment
- `assignment2_upstox_authentication.py` - Solution for Upstox authentication assignment
- `assignment3_instrument_mapping.py` - Solution for instrument mapping assignment
- `assignment4_historical_data.py` - Solution for historical data fetching assignment
- `assignment5_batch_processing.py` - Solution for batch processing assignment
- `assignment6_data_resampling.py` - Solution for data resampling assignment
- `assignment7_market_scanner.py` - Solution for market scanner assignment

## Prerequisites

To work with these materials, you should have:

1. Intermediate Python knowledge
2. Python 3.6+ installed
3. pandas, numpy, requests libraries
4. Basic understanding of financial markets
5. Completed Module 2 materials
6. Upstox API key and secret (free to generate at developer.upstox.com)

## Getting Started

1. Start by reading the notes in numerical order
2. After each topic, attempt the corresponding assignment
3. Compare your solution with the provided one in the solutions directory
4. Proceed to the next topic once you feel comfortable

## Learning Path

This module follows a progressive learning path:

1. **Foundation (Upstox API Basics)**
   - Understanding Upstox v3 API structure
   - OAuth 2.0 Authentication and tokens
   - Rate limiting and best practices

2. **Symbol Mapping**
   - Exchange symbols and instrument keys
   - Handling different exchanges
   - Symbol search and validation

3. **Data Fetching**
   - Historical candle data
   - Different timeframes
   - Error handling and retries

4. **Advanced Data Processing**
   - Batch processing multiple symbols
   - Working with different timeframes
   - Data resampling techniques

5. **Market Scanning**
   - Processing multiple data files
   - Implementing technical indicators
   - Creating market scanners

## Data Files

The assignments use data files from the `data` directory. Sample files are included, but most exercises will involve fetching your own data through the APIs.

## Best Practices

- Always handle API rate limits properly
- Use config files for reusable settings
- Create modular code with clear separation of concerns
- Implement proper error handling for API requests
- Use batching for multiple symbols
- Document your code with comments
- Consider performance when processing large datasets

Happy learning and trading!
