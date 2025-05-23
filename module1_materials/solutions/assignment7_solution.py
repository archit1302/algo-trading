"""
Assignment 7 Solution: Personal Trading Bot - Capstone Project
Complete solution for automated trading system using all Python concepts

Author: GitHub Copilot
Module: 1 - Python Fundamentals
Assignment: 7 - Capstone Project
"""

import csv
import json
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class OrderType(Enum):
    BUY = "BUY"
    SELL = "SELL"

class OrderStatus(Enum):
    PENDING = "PENDING"
    EXECUTED = "EXECUTED"
    CANCELLED = "CANCELLED"

@dataclass
class TradingSignal:
    symbol: str
    action: OrderType
    price: float
    quantity: int
    confidence: float
    timestamp: datetime
    reason: str

@dataclass
class Order:
    order_id: str
    symbol: str
    order_type: OrderType
    quantity: int
    price: float
    status: OrderStatus
    timestamp: datetime
    executed_price: Optional[float] = None
    executed_time: Optional[datetime] = None

@dataclass
class Position:
    symbol: str
    quantity: int
    avg_price: float
    current_price: float
    pnl: float
    pnl_percent: float

class TradingBot:
    """
    Personal Trading Bot with automated decision making
    """
    
    def __init__(self, initial_capital: float = 100000, max_position_size: float = 0.3):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_position_size = max_position_size  # Maximum 30% per position
        
        # Data storage
        self.market_data = {}
        self.positions = {}
        self.orders = []
        self.trade_history = []
        self.signals_history = []
        
        # Trading parameters
        self.stop_loss_percent = 5.0  # 5% stop loss
        self.take_profit_percent = 10.0  # 10% take profit
        self.min_confidence = 0.6  # Minimum signal confidence
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.max_drawdown = 0
        self.peak_capital = initial_capital
        
        # Setup directories
        self.data_dir = "trading_bot_data"
        self.ensure_directories()
    
    def ensure_directories(self):
        """Create necessary directories"""
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "logs"), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "reports"), exist_ok=True)
    
    # Data Management
    def load_market_data(self, filename: str) -> bool:
        """Load market data from CSV file"""
        filepath = os.path.join(self.data_dir, filename)
        
        try:
            with open(filepath, 'r', newline='') as file:
                reader = csv.DictReader(file)
                
                for row in reader:
                    symbol = row['symbol'].upper()
                    if symbol not in self.market_data:
                        self.market_data[symbol] = []
                    
                    data_point = {
                        'date': row['date'],
                        'open': float(row['open']),
                        'high': float(row['high']),
                        'low': float(row['low']),
                        'close': float(row['close']),
                        'volume': int(row['volume'])
                    }
                    self.market_data[symbol].append(data_point)
                
                # Sort by date
                for symbol in self.market_data:
                    self.market_data[symbol].sort(key=lambda x: x['date'])
                
                self.log(f"Loaded market data for {len(self.market_data)} symbols")
                return True
                
        except Exception as e:
            self.log(f"Error loading market data: {e}")
            return False
    
    def get_current_price(self, symbol: str) -> float:
        """Get latest price for symbol"""
        if symbol in self.market_data and self.market_data[symbol]:
            return self.market_data[symbol][-1]['close']
        return 0.0
    
    def get_price_history(self, symbol: str, days: int = 20) -> List[Dict]:
        """Get price history for specified days"""
        if symbol not in self.market_data:
            return []
        return self.market_data[symbol][-days:] if len(self.market_data[symbol]) >= days else self.market_data[symbol]
    
    # Technical Analysis
    def calculate_sma(self, symbol: str, period: int = 20) -> float:
        """Calculate Simple Moving Average"""
        history = self.get_price_history(symbol, period)
        if len(history) < period:
            return 0.0
        
        prices = [day['close'] for day in history[-period:]]
        return sum(prices) / len(prices)
    
    def calculate_rsi(self, symbol: str, period: int = 14) -> float:
        """Calculate Relative Strength Index"""
        history = self.get_price_history(symbol, period + 1)
        if len(history) < period + 1:
            return 50.0  # Neutral RSI
        
        # Calculate price changes
        price_changes = []
        for i in range(1, len(history)):
            change = history[i]['close'] - history[i-1]['close']
            price_changes.append(change)
        
        # Separate gains and losses
        gains = [change if change > 0 else 0 for change in price_changes]
        losses = [-change if change < 0 else 0 for change in price_changes]
        
        # Calculate average gains and losses
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_bollinger_bands(self, symbol: str, period: int = 20, std_dev: float = 2.0) -> Tuple[float, float, float]:
        """Calculate Bollinger Bands (upper, middle, lower)"""
        history = self.get_price_history(symbol, period)
        if len(history) < period:
            current_price = self.get_current_price(symbol)
            return current_price, current_price, current_price
        
        prices = [day['close'] for day in history[-period:]]
        middle = sum(prices) / len(prices)
        
        # Calculate standard deviation
        variance = sum((price - middle) ** 2 for price in prices) / len(prices)
        std = variance ** 0.5
        
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        
        return upper, middle, lower
    
    def detect_price_breakout(self, symbol: str) -> Dict:
        """Detect price breakouts"""
        history = self.get_price_history(symbol, 20)
        if len(history) < 10:
            return {'breakout': False, 'direction': None, 'strength': 0}
        
        current_price = history[-1]['close']
        recent_high = max(day['high'] for day in history[-10:-1])  # Exclude current day
        recent_low = min(day['low'] for day in history[-10:-1])
        
        # Check for breakouts
        if current_price > recent_high * 1.02:  # 2% above recent high
            strength = (current_price - recent_high) / recent_high
            return {'breakout': True, 'direction': 'upward', 'strength': strength}
        elif current_price < recent_low * 0.98:  # 2% below recent low
            strength = (recent_low - current_price) / recent_low
            return {'breakout': True, 'direction': 'downward', 'strength': strength}
        
        return {'breakout': False, 'direction': None, 'strength': 0}
    
    # Signal Generation
    def generate_trading_signals(self, symbol: str) -> Optional[TradingSignal]:
        """Generate trading signals based on technical analysis"""
        current_price = self.get_current_price(symbol)
        if current_price == 0:
            return None
        
        # Technical indicators
        sma_20 = self.calculate_sma(symbol, 20)
        sma_50 = self.calculate_sma(symbol, 50)
        rsi = self.calculate_rsi(symbol)
        upper_bb, middle_bb, lower_bb = self.calculate_bollinger_bands(symbol)
        breakout = self.detect_price_breakout(symbol)
        
        # Initialize signal components
        signals = []
        reasons = []
        
        # Moving Average Crossover Strategy
        if sma_20 > 0 and sma_50 > 0:
            if sma_20 > sma_50 and current_price > sma_20:
                signals.append(('BUY', 0.7))
                reasons.append("Price above rising SMA20")
            elif sma_20 < sma_50 and current_price < sma_20:
                signals.append(('SELL', 0.7))
                reasons.append("Price below falling SMA20")
        
        # RSI Strategy
        if rsi < 30:  # Oversold
            signals.append(('BUY', 0.8))
            reasons.append(f"RSI oversold ({rsi:.1f})")
        elif rsi > 70:  # Overbought
            signals.append(('SELL', 0.8))
            reasons.append(f"RSI overbought ({rsi:.1f})")
        
        # Bollinger Bands Strategy
        if current_price < lower_bb:
            signals.append(('BUY', 0.6))
            reasons.append("Price below lower Bollinger Band")
        elif current_price > upper_bb:
            signals.append(('SELL', 0.6))
            reasons.append("Price above upper Bollinger Band")
        
        # Breakout Strategy
        if breakout['breakout']:
            if breakout['direction'] == 'upward' and breakout['strength'] > 0.03:
                signals.append(('BUY', 0.9))
                reasons.append(f"Upward breakout ({breakout['strength']*100:.1f}%)")
            elif breakout['direction'] == 'downward' and breakout['strength'] > 0.03:
                signals.append(('SELL', 0.9))
                reasons.append(f"Downward breakout ({breakout['strength']*100:.1f}%)")
        
        # Aggregate signals
        if not signals:
            return None
        
        buy_signals = [s for s in signals if s[0] == 'BUY']
        sell_signals = [s for s in signals if s[0] == 'SELL']
        
        if len(buy_signals) > len(sell_signals):
            avg_confidence = sum(s[1] for s in buy_signals) / len(buy_signals)
            action = OrderType.BUY
            relevant_reasons = [r for i, r in enumerate(reasons) if signals[i][0] == 'BUY']
        elif len(sell_signals) > len(buy_signals):
            avg_confidence = sum(s[1] for s in sell_signals) / len(sell_signals)
            action = OrderType.SELL
            relevant_reasons = [r for i, r in enumerate(reasons) if signals[i][0] == 'SELL']
        else:
            return None  # Conflicting signals
        
        # Calculate position size
        max_investment = self.current_capital * self.max_position_size
        quantity = int(max_investment / current_price)
        
        if quantity == 0:
            return None
        
        return TradingSignal(
            symbol=symbol,
            action=action,
            price=current_price,
            quantity=quantity,
            confidence=avg_confidence,
            timestamp=datetime.now(),
            reason="; ".join(relevant_reasons)
        )
    
    # Order Management
    def place_order(self, signal: TradingSignal) -> bool:
        """Place trading order based on signal"""
        if signal.confidence < self.min_confidence:
            self.log(f"Signal confidence {signal.confidence:.2f} below threshold {self.min_confidence}")
            return False
        
        # Check if we already have a position
        current_position = self.positions.get(signal.symbol, None)
        
        # Risk management checks
        if signal.action == OrderType.BUY:
            required_capital = signal.quantity * signal.price
            if required_capital > self.current_capital:
                self.log(f"Insufficient capital for {signal.symbol} BUY order")
                return False
            
            if current_position and current_position.quantity > 0:
                self.log(f"Already holding position in {signal.symbol}")
                return False
        
        elif signal.action == OrderType.SELL:
            if not current_position or current_position.quantity <= 0:
                self.log(f"No position to sell in {signal.symbol}")
                return False
        
        # Create order
        order_id = f"{signal.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        order = Order(
            order_id=order_id,
            symbol=signal.symbol,
            order_type=signal.action,
            quantity=signal.quantity,
            price=signal.price,
            status=OrderStatus.PENDING,
            timestamp=signal.timestamp
        )
        
        # Execute order immediately (simulated)
        success = self.execute_order(order)
        
        if success:
            self.orders.append(order)
            self.signals_history.append(signal)
            self.log(f"Order placed: {signal.action.value} {signal.quantity} {signal.symbol} @ ₹{signal.price:.2f}")
        
        return success
    
    def execute_order(self, order: Order) -> bool:
        """Execute pending order"""
        try:
            current_price = self.get_current_price(order.symbol)
            
            if order.order_type == OrderType.BUY:
                # Execute buy order
                cost = order.quantity * current_price
                if cost > self.current_capital:
                    order.status = OrderStatus.CANCELLED
                    return False
                
                self.current_capital -= cost
                
                # Update position
                if order.symbol in self.positions:
                    existing = self.positions[order.symbol]
                    total_quantity = existing.quantity + order.quantity
                    total_cost = (existing.quantity * existing.avg_price) + cost
                    new_avg_price = total_cost / total_quantity
                    
                    self.positions[order.symbol] = Position(
                        symbol=order.symbol,
                        quantity=total_quantity,
                        avg_price=new_avg_price,
                        current_price=current_price,
                        pnl=0,
                        pnl_percent=0
                    )
                else:
                    self.positions[order.symbol] = Position(
                        symbol=order.symbol,
                        quantity=order.quantity,
                        avg_price=current_price,
                        current_price=current_price,
                        pnl=0,
                        pnl_percent=0
                    )
            
            elif order.order_type == OrderType.SELL:
                # Execute sell order
                if order.symbol not in self.positions:
                    order.status = OrderStatus.CANCELLED
                    return False
                
                position = self.positions[order.symbol]
                if position.quantity < order.quantity:
                    order.status = OrderStatus.CANCELLED
                    return False
                
                revenue = order.quantity * current_price
                self.current_capital += revenue
                
                # Calculate P&L for this trade
                cost_basis = order.quantity * position.avg_price
                trade_pnl = revenue - cost_basis
                
                # Update position
                remaining_quantity = position.quantity - order.quantity
                if remaining_quantity == 0:
                    del self.positions[order.symbol]
                else:
                    self.positions[order.symbol].quantity = remaining_quantity
                
                # Record trade
                self.record_trade(order.symbol, order.quantity, position.avg_price, current_price, trade_pnl)
            
            # Mark order as executed
            order.status = OrderStatus.EXECUTED
            order.executed_price = current_price
            order.executed_time = datetime.now()
            
            return True
            
        except Exception as e:
            self.log(f"Error executing order {order.order_id}: {e}")
            order.status = OrderStatus.CANCELLED
            return False
    
    def record_trade(self, symbol: str, quantity: int, buy_price: float, sell_price: float, pnl: float):
        """Record completed trade"""
        trade = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'quantity': quantity,
            'buy_price': buy_price,
            'sell_price': sell_price,
            'pnl': pnl,
            'pnl_percent': (pnl / (quantity * buy_price)) * 100,
            'trade_id': len(self.trade_history) + 1
        }
        
        self.trade_history.append(trade)
        self.total_trades += 1
        
        if pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        self.log(f"Trade completed: {symbol} P&L ₹{pnl:+,.2f} ({trade['pnl_percent']:+.2f}%)")
    
    # Risk Management
    def check_stop_loss_take_profit(self):
        """Check and execute stop loss/take profit orders"""
        for symbol, position in list(self.positions.items()):
            current_price = self.get_current_price(symbol)
            
            # Update position P&L
            position.current_price = current_price
            position.pnl = (current_price - position.avg_price) * position.quantity
            position.pnl_percent = ((current_price - position.avg_price) / position.avg_price) * 100
            
            # Check stop loss
            if position.pnl_percent <= -self.stop_loss_percent:
                self.log(f"Stop loss triggered for {symbol}: {position.pnl_percent:.2f}%")
                signal = TradingSignal(
                    symbol=symbol,
                    action=OrderType.SELL,
                    price=current_price,
                    quantity=position.quantity,
                    confidence=1.0,
                    timestamp=datetime.now(),
                    reason="Stop loss triggered"
                )
                self.place_order(signal)
            
            # Check take profit
            elif position.pnl_percent >= self.take_profit_percent:
                self.log(f"Take profit triggered for {symbol}: {position.pnl_percent:.2f}%")
                signal = TradingSignal(
                    symbol=symbol,
                    action=OrderType.SELL,
                    price=current_price,
                    quantity=position.quantity,
                    confidence=1.0,
                    timestamp=datetime.now(),
                    reason="Take profit triggered"
                )
                self.place_order(signal)
    
    def calculate_portfolio_value(self) -> float:
        """Calculate total portfolio value"""
        total_value = self.current_capital
        
        for position in self.positions.values():
            current_price = self.get_current_price(position.symbol)
            position_value = position.quantity * current_price
            total_value += position_value
        
        return total_value
    
    def update_drawdown(self):
        """Update maximum drawdown"""
        current_value = self.calculate_portfolio_value()
        
        if current_value > self.peak_capital:
            self.peak_capital = current_value
        
        drawdown = (self.peak_capital - current_value) / self.peak_capital * 100
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
    
    # Bot Operation
    def run_trading_cycle(self, symbols: List[str]):
        """Run one complete trading cycle"""
        self.log("=== Starting Trading Cycle ===")
        
        # Check risk management first
        self.check_stop_loss_take_profit()
        
        # Generate signals for each symbol
        for symbol in symbols:
            try:
                signal = self.generate_trading_signals(symbol)
                if signal:
                    self.log(f"Signal generated for {symbol}: {signal.action.value} (confidence: {signal.confidence:.2f})")
                    self.log(f"Reason: {signal.reason}")
                    self.place_order(signal)
                    
                    # Small delay between orders
                    time.sleep(0.1)
                
            except Exception as e:
                self.log(f"Error processing {symbol}: {e}")
        
        # Update performance metrics
        self.update_drawdown()
        
        portfolio_value = self.calculate_portfolio_value()
        self.log(f"Portfolio Value: ₹{portfolio_value:,.2f}")
        self.log(f"Cash: ₹{self.current_capital:,.2f}")
        self.log(f"Positions: {len(self.positions)}")
        self.log("=== Trading Cycle Complete ===\n")
    
    def run_backtest(self, symbols: List[str], days: int = 30):
        """Run backtest simulation"""
        self.log(f"Starting backtest for {days} days with symbols: {', '.join(symbols)}")
        
        # Simulate trading for specified days
        for day in range(days):
            self.log(f"\n--- Day {day + 1} ---")
            self.run_trading_cycle(symbols)
            
            # Simulate next day (advance data if available)
            # In real implementation, this would load next day's data
            time.sleep(0.01)  # Small delay for simulation
        
        self.log("Backtest completed")
    
    # Reporting and Logging
    def log(self, message: str):
        """Log message with timestamp"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        
        # Save to log file
        log_file = os.path.join(self.data_dir, "logs", f"trading_log_{datetime.now().strftime('%Y%m%d')}.txt")
        with open(log_file, 'a', encoding='utf-8') as file:
            file.write(log_message + '\n')
    
    def generate_performance_report(self) -> Dict:
        """Generate comprehensive performance report"""
        portfolio_value = self.calculate_portfolio_value()
        total_return = portfolio_value - self.initial_capital
        total_return_percent = (total_return / self.initial_capital) * 100
        
        win_rate = (self.winning_trades / self.total_trades) * 100 if self.total_trades > 0 else 0
        
        avg_win = 0
        avg_loss = 0
        if self.trade_history:
            winning_trades = [t['pnl'] for t in self.trade_history if t['pnl'] > 0]
            losing_trades = [t['pnl'] for t in self.trade_history if t['pnl'] < 0]
            
            avg_win = sum(winning_trades) / len(winning_trades) if winning_trades else 0
            avg_loss = sum(losing_trades) / len(losing_trades) if losing_trades else 0
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'initial_capital': self.initial_capital,
            'current_portfolio_value': portfolio_value,
            'total_return': total_return,
            'total_return_percent': total_return_percent,
            'max_drawdown': self.max_drawdown,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'average_win': avg_win,
            'average_loss': avg_loss,
            'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else 0,
            'current_positions': len(self.positions),
            'cash_remaining': self.current_capital,
            'positions_detail': [
                {
                    'symbol': pos.symbol,
                    'quantity': pos.quantity,
                    'avg_price': pos.avg_price,
                    'current_price': pos.current_price,
                    'pnl': pos.pnl,
                    'pnl_percent': pos.pnl_percent
                }
                for pos in self.positions.values()
            ],
            'recent_trades': self.trade_history[-10:] if len(self.trade_history) > 10 else self.trade_history
        }
        
        return report
    
    def save_performance_report(self, report: Dict = None):
        """Save performance report to file"""
        if not report:
            report = self.generate_performance_report()
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"performance_report_{timestamp}.json"
        filepath = os.path.join(self.data_dir, "reports", filename)
        
        with open(filepath, 'w', encoding='utf-8') as file:
            json.dump(report, file, indent=2, default=str)
        
        self.log(f"Performance report saved: {filename}")
        return filepath
    
    def print_performance_summary(self):
        """Print performance summary to console"""
        report = self.generate_performance_report()
        
        print("\n" + "="*60)
        print("TRADING BOT PERFORMANCE SUMMARY")
        print("="*60)
        print(f"Initial Capital:      ₹{report['initial_capital']:,.2f}")
        print(f"Current Value:        ₹{report['current_portfolio_value']:,.2f}")
        print(f"Total Return:         ₹{report['total_return']:+,.2f} ({report['total_return_percent']:+.2f}%)")
        print(f"Max Drawdown:         {report['max_drawdown']:.2f}%")
        print(f"Total Trades:         {report['total_trades']}")
        print(f"Win Rate:             {report['win_rate']:.1f}%")
        print(f"Winning Trades:       {report['winning_trades']}")
        print(f"Losing Trades:        {report['losing_trades']}")
        print(f"Average Win:          ₹{report['average_win']:,.2f}")
        print(f"Average Loss:         ₹{report['average_loss']:,.2f}")
        print(f"Profit Factor:        {report['profit_factor']:.2f}")
        print(f"Current Positions:    {report['current_positions']}")
        print(f"Cash Remaining:       ₹{report['cash_remaining']:,.2f}")
        
        if report['positions_detail']:
            print(f"\nCurrent Positions:")
            for pos in report['positions_detail']:
                print(f"  {pos['symbol']}: {pos['quantity']} @ ₹{pos['avg_price']:.2f} | P&L: ₹{pos['pnl']:+,.2f} ({pos['pnl_percent']:+.2f}%)")
        
        print("="*60)

def create_sample_trading_data():
    """Create extended sample data for trading bot testing"""
    import random
    
    symbols = ['SBIN', 'RELIANCE', 'TCS', 'INFY', 'HDFC']
    base_prices = {'SBIN': 850, 'RELIANCE': 2450, 'TCS': 3200, 'INFY': 1450, 'HDFC': 1580}
    
    trading_data = []
    start_date = datetime(2024, 1, 1)
    
    for symbol in symbols:
        current_price = base_prices[symbol]
        
        # Generate 30 days of data
        for day in range(30):
            date = start_date + timedelta(days=day)
            
            # Simulate price movement
            change_percent = random.uniform(-0.05, 0.05)  # ±5% daily change
            price_change = current_price * change_percent
            
            # OHLC calculation
            open_price = current_price
            close_price = current_price + price_change
            
            high_price = max(open_price, close_price) * random.uniform(1.0, 1.03)
            low_price = min(open_price, close_price) * random.uniform(0.97, 1.0)
            
            volume = random.randint(1000000, 5000000)
            
            trading_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'symbol': symbol,
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
                'volume': volume
            })
            
            current_price = close_price
    
    return trading_data

def main():
    """Demonstrate the Personal Trading Bot"""
    print("=== Assignment 7: Personal Trading Bot - Capstone Project ===\n")
    
    # Initialize trading bot
    print("1. Initializing Trading Bot...")
    bot = TradingBot(initial_capital=500000, max_position_size=0.25)  # ₹5 lakh capital, 25% max position
    print(f"Bot initialized with ₹{bot.initial_capital:,.2f} capital")
    print()
    
    # Create and save sample trading data
    print("2. Generating sample market data...")
    sample_data = create_sample_trading_data()
    
    # Save data to CSV
    data_file = os.path.join(bot.data_dir, 'trading_data.csv')
    with open(data_file, 'w', newline='') as file:
        fieldnames = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sample_data)
    
    print(f"Generated {len(sample_data)} data points for trading")
    print()
    
    # Load market data
    print("3. Loading market data...")
    bot.load_market_data('trading_data.csv')
    print()
    
    # Configure trading parameters
    print("4. Configuring trading parameters...")
    bot.stop_loss_percent = 3.0  # 3% stop loss
    bot.take_profit_percent = 8.0  # 8% take profit
    bot.min_confidence = 0.6  # 60% minimum confidence
    print(f"Stop Loss: {bot.stop_loss_percent}%, Take Profit: {bot.take_profit_percent}%")
    print(f"Minimum Signal Confidence: {bot.min_confidence}")
    print()
    
    # Test signal generation
    print("5. Testing signal generation...")
    test_symbols = ['SBIN', 'RELIANCE', 'TCS']
    for symbol in test_symbols:
        signal = bot.generate_trading_signals(symbol)
        if signal:
            print(f"{symbol}: {signal.action.value} {signal.quantity} shares @ ₹{signal.price:.2f}")
            print(f"  Confidence: {signal.confidence:.2f}, Reason: {signal.reason}")
        else:
            print(f"{symbol}: No signal generated")
    print()
    
    # Run trading simulation
    print("6. Running trading simulation...")
    symbols_to_trade = ['SBIN', 'RELIANCE', 'TCS']
    
    # Run several trading cycles
    for cycle in range(5):
        print(f"\n--- Trading Cycle {cycle + 1} ---")
        bot.run_trading_cycle(symbols_to_trade)
        time.sleep(0.5)  # Pause between cycles
    
    print()
    
    # Generate and display performance report
    print("7. Generating performance report...")
    bot.print_performance_summary()
    
    # Save detailed report
    report_file = bot.save_performance_report()
    print(f"\nDetailed report saved to: {report_file}")
    
    # Show bot capabilities
    print("\n=== Trading Bot Capabilities Demonstrated ===")
    print("✓ Market data loading and processing")
    print("✓ Technical analysis (SMA, RSI, Bollinger Bands, Breakouts)")
    print("✓ Automated signal generation")
    print("✓ Risk management (Stop Loss, Take Profit)")
    print("✓ Order placement and execution")
    print("✓ Portfolio management")
    print("✓ Performance tracking and reporting")
    print("✓ Comprehensive logging")
    print("✓ Data export and analysis")
    
    print("\n=== Capstone Project Complete ===")
    print("The Personal Trading Bot successfully integrates all Python concepts:")
    print("• Variables and data types for market data")
    print("• Lists and dictionaries for data structures")
    print("• Control flow for trading logic")
    print("• Functions for modular code organization")
    print("• File handling for data persistence")
    print("• Classes and objects for system architecture")
    print("• Error handling for robust operation")
    print("• Real-world application in financial markets")

if __name__ == "__main__":
    main()
