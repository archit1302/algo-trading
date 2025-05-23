import ccxt
import pandas as pd
from strategies.base_strategy import BaseStrategy

class EMALongStrategy(BaseStrategy):
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
            df['ema10'] = df['close'].ewm(span=self.params.get('ema_fast', 10),  adjust=False).mean()
            df['ema30'] = df['close'].ewm(span=self.params.get('ema_slow', 30),  adjust=False).mean()
            return df
        except Exception as e:
            self.logger.error(f"Error fetching {symbol}: {e}")
            return None

    def generate_signal(self, symbol: str, df):
        # need at least 3 bars: prev crossover bar and next bar for signal
        if df is None or len(df) < 3:
            return None

        prev = df.iloc[-3]   # bar before potential crossover
        curr = df.iloc[-2]   # bar where crossover must occur
        next_open = df.iloc[-1]['open']  # entry price on the next bar's open

        # 1) 10 EMA crosses above 30 EMA on curr
        crossed_up   = (prev['ema10'] <= prev['ema30']) and (curr['ema10'] > curr['ema30'])
        # 2) that same bar closes above both EMAs
        close_above  = (curr['close'] > curr['ema10']) and (curr['close'] > curr['ema30'])

        if crossed_up and close_above:
            return {
                'strategy':       self.name,
                'symbol':         symbol,
                'price':          next_open,
                'side':           'buy',
                'quantity_inr':   self.quantity_inr,
                'leverage':       self.leverage,
                'stop_loss_pct':  self.stop_loss_pct,
                'take_profit_pct':self.take_profit_pct
            }
        return None
