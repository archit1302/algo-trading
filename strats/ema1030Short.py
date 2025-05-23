import ccxt
import pandas as pd
from strategies.base_strategy import BaseStrategy

class EMAShortStrategy(BaseStrategy):
    def __init__(self, config, order_manager, notifier, logger):
        super().__init__(config, order_manager, notifier, logger)
        self.exchange = getattr(ccxt, config['exchange'])()
        self.timeframe = config['timeframe']
        self.limit = config['limit']

    def fetch_data(self, symbol: str):
        try:
            bars = self.exchange.fetch_ohlcv(symbol, timeframe=self.timeframe, limit=self.limit)
            df = pd.DataFrame(bars, columns=['timestamp','open','high','low','close','volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            # EMAs: default to 10 & 30 but overridable via params
            df['ema10'] = df['close'].ewm(span=self.params.get('ema_fast', 10), adjust=False).mean()
            df['ema30'] = df['close'].ewm(span=self.params.get('ema_slow', 30), adjust=False).mean()
            return df
        except Exception as e:
            self.logger.error(f"Error fetching {symbol}: {e}")
            return None

    def generate_signal(self, symbol: str, df):
        # need at least 3 bars: bar before crossover, crossover bar, and next bar for entry
        if df is None or len(df) < 3:
            return None

        prev = df.iloc[-3]                 # bar before potential crossover
        curr = df.iloc[-2]                 # bar where crossover must occur
        next_open = df.iloc[-1]['open']    # entry price on the next bar's open

        # 1) 10 EMA crosses below 30 EMA on curr
        crossed_down = (prev['ema10'] >= prev['ema30']) and (curr['ema10'] < curr['ema30'])
        # 2) that same bar closes below both EMAs
        close_below   = (curr['close'] < curr['ema10']) and (curr['close'] < curr['ema30'])

        if crossed_down and close_below:
            return {
                'strategy':        self.name,
                'symbol':          symbol,
                'price':           next_open,
                'side':            'sell',
                'quantity_inr':    self.quantity_inr,
                'leverage':        self.leverage,
                'stop_loss_pct':   self.stop_loss_pct,
                'take_profit_pct': self.take_profit_pct
            }
        return None
