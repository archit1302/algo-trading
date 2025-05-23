# Introduction to Upstox API v3

This note provides an introduction to the Upstox API v3, including its features, architecture, and basic usage patterns.

## What is Upstox API?

Upstox API is a REST-based platform that enables developers to build trading applications that can access market data, place orders, and manage portfolios on the Upstox platform. The API v3 is the latest version, offering enhanced performance and additional features compared to previous versions.

## Key Features of Upstox API v3

1. **Historical Data Access**: Fetch historical candle data for multiple timeframes and instruments.
2. **Real-time Market Data**: Access live market data through WebSockets.
3. **Order Management**: Place, modify, and cancel orders.
4. **Portfolio Management**: View holdings, positions, and order history.
5. **User Profile**: Access user account information.
6. **Custom Timeframes**: Get data for custom intervals (minute, hour, day, week, month).

## API Architecture

The Upstox API v3 follows a standard REST architecture with the following components:

1. **Authentication**: OAuth 2.0 based authentication flow
2. **Endpoints**: HTTP endpoints for different functionalities
3. **Request/Response Format**: JSON-based request and response format
4. **Rate Limiting**: Limitations on the number of requests per minute
5. **WebSockets**: For real-time market data streaming

## Base URL

The base URL for all API v3 endpoints is:

```
https://api.upstox.com/v3
```

## Authentication Flow

Upstox uses OAuth 2.0 for authentication. The flow involves:

1. **Authorization URL**: Direct users to the Upstox login page
2. **Authorization Code**: Receive an authorization code after successful login
3. **Access Token**: Exchange the authorization code for an access token
4. **API Requests**: Use the access token for all subsequent API requests

### Example Authentication Flow

```
┌─────────────┐          ┌───────────┐          ┌───────────┐
│             │          │           │          │           │
│  Your App   │          │  Browser  │          │  Upstox   │
│             │          │           │          │           │
└──────┬──────┘          └─────┬─────┘          └─────┬─────┘
       │                       │                      │
       │  1. Generate Auth URL │                      │
       │─────────────────────>│                      │
       │                       │  2. Open Auth URL    │
       │                       │─────────────────────>│
       │                       │                      │
       │                       │  3. User Login       │
       │                       │<─────────────────────│
       │                       │                      │
       │                       │  4. Grant Access     │
       │                       │─────────────────────>│
       │                       │                      │
       │                       │  5. Redirect w/ Code │
       │                       │<─────────────────────│
       │  6. Code Callback     │                      │
       │<──────────────────────│                      │
       │                       │                      │
       │  7. Exchange Code for │                      │
       │     Access Token      │                      │
       │─────────────────────────────────────────────>│
       │                       │                      │
       │  8. Access Token      │                      │
       │<─────────────────────────────────────────────│
       │                       │                      │
```

## Rate Limits

Upstox API has rate limits to prevent abuse and ensure fair usage:

| API Category | Rate Limit |
|--------------|------------|
| Market Data  | 180 requests per minute |
| Order APIs   | 60 requests per minute |
| Other APIs   | 120 requests per minute |

When rate limits are exceeded, the API returns a 429 status code with a `Retry-After` header indicating the number of seconds to wait before making another request.

## Key API Endpoints

### 1. Historical Data

```
GET /historical-candle/{instrument_key}/{unit}/{interval}
```

Parameters:
- `instrument_key`: Unique identifier for the instrument
- `unit`: Time unit (minute, hour, day, week, month)
- `interval`: Number of units per candle (1, 2, 3, etc.)
- `from_date`: Start date (YYYY-MM-DD)
- `to_date`: End date (YYYY-MM-DD)

Example Response:
```json
{
  "status": "success",
  "data": {
    "candles": [
      [
        "2023-01-01T09:15:00+05:30",
        540.5,
        544.7,
        539.0,
        542.3,
        1254867,
        0
      ],
      [
        "2023-01-01T09:30:00+05:30",
        542.4,
        545.2,
        541.8,
        544.1,
        987654,
        0
      ]
    ]
  }
}
```

### 2. Instruments

Instrument data is available as downloadable files:

- **Complete**: `https://assets.upstox.com/market-quote/instruments/exchange/complete.json.gz`
- **NSE**: `https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz`
- **BSE**: `https://assets.upstox.com/market-quote/instruments/exchange/BSE.json.gz`
- **MCX**: `https://assets.upstox.com/market-quote/instruments/exchange/MCX.json.gz`

JSON format example:
```json
{
  "segment": "NSE_EQ",
  "name": "STATE BANK OF INDIA",
  "exchange": "NSE",
  "isin": "INE062A01020",
  "instrument_type": "EQ",
  "instrument_key": "NSE_EQ|INE062A01020",
  "lot_size": 1,
  "freeze_quantity": 250000.0,
  "exchange_token": "3045",
  "tick_size": 5.0,
  "trading_symbol": "SBIN",
  "short_name": "SBIN",
  "security_type": "NORMAL"
}
```

## Best Practices

1. **Handle Rate Limits**: Implement exponential backoff when rate limits are hit
2. **Use Instrument Keys**: Always use `instrument_key` for uniquely identifying instruments
3. **Error Handling**: Properly handle API errors and implement retries
4. **Token Management**: Safely store and refresh access tokens
5. **Batch Requests**: Group requests together where possible to minimize API calls
6. **Date Ranges**: Be mindful of date range limitations for different timeframes

## Common Error Codes

| Status Code | Description | Handling Strategy |
|-------------|-------------|-------------------|
| 400 | Bad Request | Check request parameters |
| 401 | Unauthorized | Refresh authentication token |
| 403 | Forbidden | Check API permissions |
| 404 | Not Found | Verify endpoint and parameters |
| 429 | Too Many Requests | Implement backoff and retry |
| 500 | Internal Server Error | Retry after delay |

## Tools and Libraries

Several tools and libraries can help you work with Upstox API:

1. **Python Requests**: For making HTTP requests
2. **Pandas**: For data manipulation and analysis
3. **WebSocket-client**: For connecting to WebSocket streams
4. **TA-Lib**: For technical analysis on historical data

## Example: Basic API Call

Here's a simple example of making an API call to get historical data:

```python
import requests
import pandas as pd
from datetime import datetime, timedelta

def get_historical_data(access_token, instrument_key, interval=1, unit="minute"):
    """Fetch historical candle data from Upstox API v3"""
    
    # API endpoint
    url = f"https://api.upstox.com/v3/historical-candle/{instrument_key}/{unit}/{interval}"
    
    # Set up headers with access token
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {access_token}"
    }
    
    # Set up parameters
    to_date = datetime.now().strftime("%Y-%m-%d")
    from_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    
    params = {
        "to_date": to_date,
        "from_date": from_date
    }
    
    # Make the API request
    response = requests.get(url, headers=headers, params=params)
    
    # Check if request was successful
    if response.status_code == 200:
        data = response.json()
        
        # Extract candle data
        candles = data.get('data', {}).get('candles', [])
        
        # Convert to DataFrame
        df = pd.DataFrame(
            candles,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi']
        )
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        return df
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None
```

## Conclusion

Upstox API v3 provides a powerful platform for building trading applications with access to market data and trading functionality. By understanding the API architecture, authentication process, and best practices, you can build robust applications that leverage the full capabilities of the Upstox platform.

In the next note, we'll dive deeper into the authentication process for Upstox API v3.
