import ccxt
import pandas as pd
from strategies.base_strategy import BaseStrategy

class ReversalShortStrategy(BaseStrategy):
    def __init__(self, config, order_manager, notifier, logger):
        super().__init__(config, order_manager, notifier, logger)
        self.exchange = getattr(ccxt, config['exchange'])()
        self.timeframe = config['timeframe']
        self.limit = config['limit']
        # Bollinger params
        self.bb_len = config['params'].get('bb_len', 20)
        self.bb_dev = config['params'].get('bb_dev', 2.0)
        # RSI param
        self.rsi_len = config['params'].get('rsi_len', 14)

    def fetch_data(self, symbol: str):
        try:
            bars = self.exchange.fetch_ohlcv(symbol, timeframe=self.timeframe, limit=self.limit)
            df = pd.DataFrame(bars, columns=['timestamp','open','high','low','close','volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            # Bollinger Bands
            m   = df['close'].rolling(self.bb_len).mean()
            std = df['close'].rolling(self.bb_len).std()
            df['bb_mid']   = m
            df['bb_upper'] = m + std * self.bb_dev
            df['bb_lower'] = m - std * self.bb_dev
            # RSI
            delta = df['close'].diff()
            gain  = delta.clip(lower=0)
            loss  = -delta.clip(upper=0)
            avg_gain = gain.rolling(self.rsi_len).mean()
            avg_loss = loss.rolling(self.rsi_len).mean()
            rs = avg_gain / avg_loss
            df['rsi'] = 100 - (100 / (1 + rs))
            return df
        except Exception as e:
            self.logger.error(f"Error fetching {symbol} for ReversalShort: {e}")
            return None

    def generate_signal(self, symbol: str, df):
        # need at least bb_len + rsi_len + 3 bars
        min_bars = max(self.bb_len, self.rsi_len) + 3
        if df is None or len(df) < min_bars:
            return None

        trigger_bar      = df.iloc[-3]     # bar where price spiked outside upper BB
        confirmation_bar = df.iloc[-2]     # next bar for confirmation
        entry_open       = df.iloc[-1]['open']

        
        # ==== ENTRY conditions ====
        # 1) trigger bar high > upper BB and close < lower BB
        cond1 = (trigger_bar['high']  > trigger_bar['bb_upper']) and \
                (trigger_bar['close'] < trigger_bar['bb_upper'])
        # 2) confirmation bar close & high both below upper BB (inside band)
        cond2 = (confirmation_bar['close'] < trigger_bar['bb_upper']) and \
                (confirmation_bar['high'] < trigger_bar['bb_upper'])
        # 3) BB width % > 1% on trigger bar
        width_pct = ((trigger_bar['bb_upper'] - trigger_bar['bb_lower']) 
                     / trigger_bar['bb_lower']) * 100
        cond3 = width_pct > 1
        # 4) RSI on confirmation bar > 60
        cond4 = confirmation_bar['rsi'] > 60

        if all([cond1, cond2, cond3, cond4]):
            return {
                'strategy':        self.name,
                'symbol':          symbol,
                'price':           entry_open,
                'side':            'sell',
                'quantity_inr':    self.quantity_inr,
                'leverage':        self.leverage,
                'stop_loss_pct':   self.stop_loss_pct,
                'take_profit_pct': self.take_profit_pct
            }

        return None
