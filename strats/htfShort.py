import ccxt
import pandas as pd
from strategies.base_strategy import BaseStrategy

class EMAHTFShortStrategy(BaseStrategy):
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
            # 50- and 200-period EMAs (overridable via params)
            df['ema50']  = df['close'].ewm(span=self.params.get('ema_mid', 50),  adjust=False).mean()
            df['ema200'] = df['close'].ewm(span=self.params.get('ema_long', 200), adjust=False).mean()
            return df
        except Exception as e:
            self.logger.error(f"Error fetching {symbol}: {e}")
            return None

    def generate_signal(self, symbol: str, df):
        # need at least 3 bars: cross bar, follow-through bar, and next bar for entry/exit
        if df is None or len(df) < 3:
            return None

        cross_bar  = df.iloc[-3]                  # bar where 50 EMA cross must occur
        follow_bar = df.iloc[-2]                  # next bar: must close below cross_bar.low
        entry_open = df.iloc[-1]['open']          # price for entry/exit on this bar’s open

        # ==== ENTRY conditions ====
        # 1) cross_bar open above 50 EMA & close below 50 EMA
        crossed_50    = (cross_bar['open'] > cross_bar['ema50']) and (cross_bar['close'] < cross_bar['ema50'])
        # 2) entire setup below 200 EMA
        below_200     = (cross_bar['close'] < cross_bar['ema200']) and (follow_bar['close'] < follow_bar['ema200'])
        # 3) follow-through bar closes below low of cross_bar
        follow_break  = follow_bar['close'] < cross_bar['low']

        if crossed_50 and below_200 and follow_break:
            return {
                'strategy':        self.name,
                'symbol':          symbol,
                'price':           entry_open,
                'side':            'sell',     # open short
                'quantity_inr':    self.quantity_inr,
                'leverage':        self.leverage,
                'stop_loss_pct':   self.stop_loss_pct,
                'take_profit_pct': self.take_profit_pct
            }

        return None
