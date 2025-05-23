"""
Assignment 6 Solution: Integrated Financial Data Analysis System
Complete solution combining all concepts for comprehensive market analysis

Author: GitHub Copilot
Module: 1 - Python Fundamentals
Assignment: 6 - Integration Project
"""

import csv
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional

class FinancialAnalysisSystem:
    """
    Comprehensive financial analysis system integrating all Python concepts
    """
    
    def __init__(self, data_directory: str = "financial_data"):
        self.data_directory = data_directory
        self.stock_data = {}
        self.portfolio = {}
        self.analysis_results = {}
        self.ensure_data_directory()
    
    def ensure_data_directory(self):
        """Create data directory if it doesn't exist"""
        if not os.path.exists(self.data_directory):
            os.makedirs(self.data_directory)
            print(f"Created {self.data_directory} directory")
    
    # Data Loading and Management
    def load_stock_data(self, filename: str) -> bool:
        """
        Load stock data from CSV file
        
        Args:
            filename (str): CSV file containing stock data
            
        Returns:
            bool: Success status
        """
        filepath = os.path.join(self.data_directory, filename)
        
        try:
            with open(filepath, 'r', newline='', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                
                for row in reader:
                    symbol = row['symbol'].upper()
                    date = row['date']
                    
                    # Initialize symbol if not exists
                    if symbol not in self.stock_data:
                        self.stock_data[symbol] = []
                    
                    # Process and add data point
                    data_point = {
                        'date': date,
                        'open': float(row['open']),
                        'high': float(row['high']),
                        'low': float(row['low']),
                        'close': float(row['close']),
                        'volume': int(row['volume'])
                    }
                    self.stock_data[symbol].append(data_point)
                
                # Sort data by date for each symbol
                for symbol in self.stock_data:
                    self.stock_data[symbol].sort(key=lambda x: x['date'])
                
                print(f"Loaded data for {len(self.stock_data)} symbols from {filename}")
                return True
                
        except FileNotFoundError:
            print(f"Error: File '{filename}' not found")
            return False
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def create_portfolio(self, holdings: List[Dict]) -> bool:
        """
        Create portfolio from holdings data
        
        Args:
            holdings (List[Dict]): List of holdings with symbol, quantity, avg_price
            
        Returns:
            bool: Success status
        """
        try:
            for holding in holdings:
                symbol = holding['symbol'].upper()
                if symbol not in self.stock_data:
                    print(f"Warning: No data available for {symbol}")
                    continue
                
                self.portfolio[symbol] = {
                    'quantity': int(holding['quantity']),
                    'avg_price': float(holding['avg_price']),
                    'current_price': self.get_latest_price(symbol),
                    'investment': int(holding['quantity']) * float(holding['avg_price'])
                }
            
            print(f"Portfolio created with {len(self.portfolio)} holdings")
            return True
            
        except Exception as e:
            print(f"Error creating portfolio: {e}")
            return False
    
    # Price and Return Calculations
    def get_latest_price(self, symbol: str) -> float:
        """Get latest closing price for a symbol"""
        if symbol in self.stock_data and self.stock_data[symbol]:
            return self.stock_data[symbol][-1]['close']
        return 0.0
    
    def calculate_returns(self, symbol: str, period_days: int = 30) -> Dict:
        """
        Calculate various return metrics for a symbol
        
        Args:
            symbol (str): Stock symbol
            period_days (int): Period for return calculation
            
        Returns:
            Dict: Return metrics
        """
        if symbol not in self.stock_data or len(self.stock_data[symbol]) < 2:
            return {}
        
        data = self.stock_data[symbol]
        current_price = data[-1]['close']
        
        # Calculate daily returns
        daily_returns = []
        for i in range(1, len(data)):
            prev_close = data[i-1]['close']
            curr_close = data[i]['close']
            daily_return = (curr_close - prev_close) / prev_close
            daily_returns.append(daily_return)
        
        # Period return (if enough data)
        period_return = 0
        if len(data) >= period_days:
            period_start_price = data[-period_days]['close']
            period_return = (current_price - period_start_price) / period_start_price
        
        # Calculate volatility (standard deviation of returns)
        if daily_returns:
            mean_return = sum(daily_returns) / len(daily_returns)
            variance = sum((r - mean_return) ** 2 for r in daily_returns) / len(daily_returns)
            volatility = variance ** 0.5
        else:
            volatility = 0
        
        return {
            'current_price': current_price,
            'daily_returns': daily_returns[-10:],  # Last 10 days
            'avg_daily_return': sum(daily_returns) / len(daily_returns) if daily_returns else 0,
            'period_return': period_return,
            'volatility': volatility,
            'total_days': len(data)
        }
    
    def analyze_portfolio_performance(self) -> Dict:
        """
        Comprehensive portfolio performance analysis
        
        Returns:
            Dict: Portfolio performance metrics
        """
        if not self.portfolio:
            return {}
        
        total_investment = 0
        total_current_value = 0
        portfolio_details = []
        
        for symbol, holding in self.portfolio.items():
            current_price = self.get_latest_price(symbol)
            current_value = holding['quantity'] * current_price
            pnl = current_value - holding['investment']
            pnl_percent = (pnl / holding['investment']) * 100 if holding['investment'] > 0 else 0
            
            # Get return metrics
            returns = self.calculate_returns(symbol)
            
            holding_detail = {
                'symbol': symbol,
                'quantity': holding['quantity'],
                'avg_price': holding['avg_price'],
                'current_price': current_price,
                'investment': holding['investment'],
                'current_value': current_value,
                'pnl': pnl,
                'pnl_percent': pnl_percent,
                'volatility': returns.get('volatility', 0) * 100,  # Convert to percentage
                'allocation_percent': 0  # Will calculate after totals
            }
            portfolio_details.append(holding_detail)
            
            total_investment += holding['investment']
            total_current_value += current_value
        
        # Calculate allocation percentages
        for detail in portfolio_details:
            detail['allocation_percent'] = (detail['current_value'] / total_current_value) * 100 if total_current_value > 0 else 0
        
        # Portfolio level metrics
        total_pnl = total_current_value - total_investment
        total_pnl_percent = (total_pnl / total_investment) * 100 if total_investment > 0 else 0
        
        # Risk metrics
        portfolio_volatility = sum(detail['volatility'] * (detail['allocation_percent'] / 100) for detail in portfolio_details)
        
        return {
            'total_investment': total_investment,
            'total_current_value': total_current_value,
            'total_pnl': total_pnl,
            'total_pnl_percent': total_pnl_percent,
            'portfolio_volatility': portfolio_volatility,
            'holdings': portfolio_details,
            'best_performer': max(portfolio_details, key=lambda x: x['pnl_percent']) if portfolio_details else None,
            'worst_performer': min(portfolio_details, key=lambda x: x['pnl_percent']) if portfolio_details else None
        }
    
    # Technical Analysis Functions
    def calculate_moving_average(self, symbol: str, period: int = 20) -> List[float]:
        """
        Calculate simple moving average
        
        Args:
            symbol (str): Stock symbol
            period (int): Moving average period
            
        Returns:
            List[float]: Moving average values
        """
        if symbol not in self.stock_data:
            return []
        
        data = self.stock_data[symbol]
        if len(data) < period:
            return []
        
        moving_averages = []
        for i in range(period - 1, len(data)):
            avg = sum(data[j]['close'] for j in range(i - period + 1, i + 1)) / period
            moving_averages.append(round(avg, 2))
        
        return moving_averages
    
    def identify_trends(self, symbol: str) -> Dict:
        """
        Identify price trends and patterns
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            Dict: Trend analysis
        """
        if symbol not in self.stock_data or len(self.stock_data[symbol]) < 5:
            return {}
        
        data = self.stock_data[symbol]
        recent_data = data[-5:]  # Last 5 days
        
        # Calculate trend direction
        price_changes = []
        for i in range(1, len(recent_data)):
            change = recent_data[i]['close'] - recent_data[i-1]['close']
            price_changes.append(change)
        
        positive_days = len([c for c in price_changes if c > 0])
        negative_days = len([c for c in price_changes if c < 0])
        
        # Determine trend
        if positive_days > negative_days:
            trend = "Upward"
        elif negative_days > positive_days:
            trend = "Downward"
        else:
            trend = "Sideways"
        
        # Support and resistance levels (simplified)
        highs = [d['high'] for d in data[-20:]]  # Last 20 days
        lows = [d['low'] for d in data[-20:]]
        
        resistance = max(highs) if highs else 0
        support = min(lows) if lows else 0
        
        return {
            'trend': trend,
            'positive_days': positive_days,
            'negative_days': negative_days,
            'support_level': support,
            'resistance_level': resistance,
            'current_price': data[-1]['close'],
            'distance_from_support': ((data[-1]['close'] - support) / support) * 100 if support > 0 else 0,
            'distance_from_resistance': ((resistance - data[-1]['close']) / resistance) * 100 if resistance > 0 else 0
        }
    
    # Risk Assessment
    def assess_portfolio_risk(self) -> Dict:
        """
        Comprehensive portfolio risk assessment
        
        Returns:
            Dict: Risk metrics and recommendations
        """
        if not self.portfolio:
            return {}
        
        performance = self.analyze_portfolio_performance()
        risk_metrics = []
        
        for holding in performance['holdings']:
            symbol = holding['symbol']
            returns = self.calculate_returns(symbol)
            
            # Risk score based on volatility and allocation
            volatility = returns.get('volatility', 0) * 100
            allocation = holding['allocation_percent']
            risk_contribution = (volatility * allocation) / 100
            
            risk_metrics.append({
                'symbol': symbol,
                'volatility': volatility,
                'allocation': allocation,
                'risk_contribution': risk_contribution,
                'pnl_percent': holding['pnl_percent']
            })
        
        # Overall risk assessment
        total_risk_score = sum(rm['risk_contribution'] for rm in risk_metrics)
        avg_volatility = sum(rm['volatility'] for rm in risk_metrics) / len(risk_metrics) if risk_metrics else 0
        
        # Risk categories
        if total_risk_score < 15:
            risk_level = "Low"
            recommendation = "Conservative portfolio suitable for risk-averse investors"
        elif total_risk_score < 25:
            risk_level = "Medium"
            recommendation = "Balanced portfolio with moderate risk-reward ratio"
        else:
            risk_level = "High"
            recommendation = "Aggressive portfolio requiring careful monitoring"
        
        return {
            'overall_risk_level': risk_level,
            'total_risk_score': total_risk_score,
            'average_volatility': avg_volatility,
            'recommendation': recommendation,
            'risk_breakdown': risk_metrics,
            'highest_risk_stock': max(risk_metrics, key=lambda x: x['volatility']) if risk_metrics else None,
            'largest_position': max(risk_metrics, key=lambda x: x['allocation']) if risk_metrics else None
        }
    
    # Reporting and Export
    def generate_comprehensive_report(self) -> Dict:
        """
        Generate comprehensive analysis report
        
        Returns:
            Dict: Complete analysis report
        """
        report = {
            'report_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'portfolio_summary': self.analyze_portfolio_performance(),
            'risk_assessment': self.assess_portfolio_risk(),
            'individual_analysis': {},
            'market_overview': {}
        }
        
        # Individual stock analysis
        for symbol in self.portfolio.keys():
            report['individual_analysis'][symbol] = {
                'returns': self.calculate_returns(symbol),
                'trends': self.identify_trends(symbol),
                'moving_averages': {
                    'ma_10': self.calculate_moving_average(symbol, 10)[-1:],  # Latest MA
                    'ma_20': self.calculate_moving_average(symbol, 20)[-1:]
                }
            }
        
        # Market overview
        all_symbols = list(self.stock_data.keys())
        if all_symbols:
            market_returns = []
            for symbol in all_symbols:
                returns = self.calculate_returns(symbol, 30)
                if returns:
                    market_returns.append(returns['period_return'])
            
            if market_returns:
                report['market_overview'] = {
                    'symbols_tracked': len(all_symbols),
                    'avg_market_return': sum(market_returns) / len(market_returns),
                    'best_performer': max(all_symbols, key=lambda s: self.calculate_returns(s, 30).get('period_return', 0)),
                    'worst_performer': min(all_symbols, key=lambda s: self.calculate_returns(s, 30).get('period_return', 0))
                }
        
        return report
    
    def save_report(self, report: Dict, filename: str = None) -> bool:
        """
        Save analysis report to JSON file
        
        Args:
            report (Dict): Report data
            filename (str): Output filename (optional)
            
        Returns:
            bool: Success status
        """
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'financial_report_{timestamp}.json'
        
        filepath = os.path.join(self.data_directory, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as file:
                json.dump(report, file, indent=2, default=str)
            print(f"Report saved to {filename}")
            return True
        except Exception as e:
            print(f"Error saving report: {e}")
            return False
    
    def export_portfolio_csv(self, filename: str = None) -> bool:
        """
        Export portfolio summary to CSV
        
        Args:
            filename (str): Output filename
            
        Returns:
            bool: Success status
        """
        performance = self.analyze_portfolio_performance()
        if not performance or not performance['holdings']:
            print("No portfolio data to export")
            return False
        
        if not filename:
            filename = f'portfolio_summary_{datetime.now().strftime("%Y%m%d")}.csv'
        
        filepath = os.path.join(self.data_directory, filename)
        
        try:
            with open(filepath, 'w', newline='', encoding='utf-8') as file:
                fieldnames = ['symbol', 'quantity', 'avg_price', 'current_price', 
                             'investment', 'current_value', 'pnl', 'pnl_percent', 
                             'allocation_percent', 'volatility']
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                
                for holding in performance['holdings']:
                    writer.writerow({
                        'symbol': holding['symbol'],
                        'quantity': holding['quantity'],
                        'avg_price': round(holding['avg_price'], 2),
                        'current_price': round(holding['current_price'], 2),
                        'investment': round(holding['investment'], 2),
                        'current_value': round(holding['current_value'], 2),
                        'pnl': round(holding['pnl'], 2),
                        'pnl_percent': round(holding['pnl_percent'], 2),
                        'allocation_percent': round(holding['allocation_percent'], 2),
                        'volatility': round(holding['volatility'], 2)
                    })
            
            print(f"Portfolio exported to {filename}")
            return True
            
        except Exception as e:
            print(f"Error exporting portfolio: {e}")
            return False

def create_sample_system_data():
    """Create comprehensive sample data for the system"""
    # Extended stock data
    sample_data = [
        # SBIN data
        {'date': '2024-01-01', 'symbol': 'SBIN', 'open': 845.00, 'high': 850.50, 'low': 840.25, 'close': 848.75, 'volume': 2500000},
        {'date': '2024-01-02', 'symbol': 'SBIN', 'open': 849.00, 'high': 855.30, 'low': 845.80, 'close': 852.40, 'volume': 2750000},
        {'date': '2024-01-03', 'symbol': 'SBIN', 'open': 851.50, 'high': 858.90, 'low': 849.20, 'close': 856.15, 'volume': 3100000},
        {'date': '2024-01-04', 'symbol': 'SBIN', 'open': 855.25, 'high': 860.75, 'low': 850.30, 'close': 853.80, 'volume': 2890000},
        {'date': '2024-01-05', 'symbol': 'SBIN', 'open': 854.10, 'high': 862.45, 'low': 851.65, 'close': 859.90, 'volume': 3250000},
        {'date': '2024-01-08', 'symbol': 'SBIN', 'open': 860.25, 'high': 867.80, 'low': 857.30, 'close': 865.45, 'volume': 2980000},
        {'date': '2024-01-09', 'symbol': 'SBIN', 'open': 866.00, 'high': 872.15, 'low': 862.80, 'close': 869.20, 'volume': 3150000},
        {'date': '2024-01-10', 'symbol': 'SBIN', 'open': 868.50, 'high': 875.90, 'low': 865.75, 'close': 872.35, 'volume': 2890000},
        
        # RELIANCE data
        {'date': '2024-01-01', 'symbol': 'RELIANCE', 'open': 2445.00, 'high': 2460.30, 'low': 2438.75, 'close': 2456.20, 'volume': 1500000},
        {'date': '2024-01-02', 'symbol': 'RELIANCE', 'open': 2458.50, 'high': 2475.80, 'low': 2450.40, 'close': 2468.90, 'volume': 1650000},
        {'date': '2024-01-03', 'symbol': 'RELIANCE', 'open': 2470.25, 'high': 2485.60, 'low': 2462.35, 'close': 2478.45, 'volume': 1820000},
        {'date': '2024-01-04', 'symbol': 'RELIANCE', 'open': 2476.80, 'high': 2490.15, 'low': 2465.25, 'close': 2472.35, 'volume': 1750000},
        {'date': '2024-01-05', 'symbol': 'RELIANCE', 'open': 2474.60, 'high': 2488.90, 'low': 2470.80, 'close': 2485.25, 'volume': 1900000},
        {'date': '2024-01-08', 'symbol': 'RELIANCE', 'open': 2487.30, 'high': 2495.75, 'low': 2480.50, 'close': 2492.80, 'volume': 1780000},
        {'date': '2024-01-09', 'symbol': 'RELIANCE', 'open': 2494.15, 'high': 2502.40, 'low': 2488.60, 'close': 2498.25, 'volume': 1650000},
        {'date': '2024-01-10', 'symbol': 'RELIANCE', 'open': 2499.80, 'high': 2508.90, 'low': 2495.30, 'close': 2505.15, 'volume': 1720000},
        
        # TCS data
        {'date': '2024-01-01', 'symbol': 'TCS', 'open': 3220.00, 'high': 3235.80, 'low': 3210.50, 'close': 3228.45, 'volume': 890000},
        {'date': '2024-01-02', 'symbol': 'TCS', 'open': 3230.25, 'high': 3245.60, 'low': 3222.80, 'close': 3238.90, 'volume': 920000},
        {'date': '2024-01-03', 'symbol': 'TCS', 'open': 3240.50, 'high': 3258.75, 'low': 3235.20, 'close': 3251.30, 'volume': 1050000},
        {'date': '2024-01-04', 'symbol': 'TCS', 'open': 3252.80, 'high': 3268.40, 'low': 3245.90, 'close': 3261.85, 'volume': 980000},
        {'date': '2024-01-05', 'symbol': 'TCS', 'open': 3263.20, 'high': 3278.50, 'low': 3256.75, 'close': 3275.60, 'volume': 1120000},
        {'date': '2024-01-08', 'symbol': 'TCS', 'open': 3277.85, 'high': 3290.20, 'low': 3270.40, 'close': 3285.95, 'volume': 1080000},
        {'date': '2024-01-09', 'symbol': 'TCS', 'open': 3287.50, 'high': 3298.75, 'low': 3280.30, 'close': 3292.40, 'volume': 950000},
        {'date': '2024-01-10', 'symbol': 'TCS', 'open': 3294.80, 'high': 3308.90, 'low': 3288.60, 'close': 3302.15, 'volume': 1020000}
    ]
    
    # Sample portfolio holdings
    sample_portfolio = [
        {'symbol': 'SBIN', 'quantity': 100, 'avg_price': 850.00},
        {'symbol': 'RELIANCE', 'quantity': 50, 'avg_price': 2460.00},
        {'symbol': 'TCS', 'quantity': 30, 'avg_price': 3240.00}
    ]
    
    return sample_data, sample_portfolio

def main():
    """Demonstrate the integrated financial analysis system"""
    print("=== Assignment 6: Integrated Financial Data Analysis System ===\n")
    
    # Initialize system
    print("1. Initializing Financial Analysis System...")
    system = FinancialAnalysisSystem()
    print()
    
    # Create and save sample data
    print("2. Creating sample market data...")
    sample_data, sample_portfolio = create_sample_system_data()
    
    # Save sample data to CSV
    with open(os.path.join(system.data_directory, 'market_data.csv'), 'w', newline='') as file:
        fieldnames = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sample_data)
    print("Sample market data saved to market_data.csv")
    print()
    
    # Load data into system
    print("3. Loading market data...")
    system.load_stock_data('market_data.csv')
    print()
    
    # Create portfolio
    print("4. Creating portfolio...")
    system.create_portfolio(sample_portfolio)
    print()
    
    # Portfolio performance analysis
    print("5. Analyzing portfolio performance...")
    performance = system.analyze_portfolio_performance()
    
    print(f"Total Investment: ₹{performance['total_investment']:,.2f}")
    print(f"Current Value: ₹{performance['total_current_value']:,.2f}")
    print(f"Total P&L: ₹{performance['total_pnl']:+,.2f} ({performance['total_pnl_percent']:+.2f}%)")
    print(f"Best Performer: {performance['best_performer']['symbol']} ({performance['best_performer']['pnl_percent']:+.2f}%)")
    print(f"Worst Performer: {performance['worst_performer']['symbol']} ({performance['worst_performer']['pnl_percent']:+.2f}%)")
    print()
    
    # Risk assessment
    print("6. Assessing portfolio risk...")
    risk = system.assess_portfolio_risk()
    print(f"Risk Level: {risk['overall_risk_level']}")
    print(f"Risk Score: {risk['total_risk_score']:.2f}")
    print(f"Average Volatility: {risk['average_volatility']:.2f}%")
    print(f"Recommendation: {risk['recommendation']}")
    print()
    
    # Individual stock analysis
    print("7. Individual stock analysis...")
    for symbol in ['SBIN', 'RELIANCE']:
        returns = system.calculate_returns(symbol)
        trends = system.identify_trends(symbol)
        print(f"\n{symbol}:")
        print(f"  Current Price: ₹{returns['current_price']:.2f}")
        print(f"  Volatility: {returns['volatility']*100:.2f}%")
        print(f"  Trend: {trends['trend']}")
        print(f"  Support Level: ₹{trends['support_level']:.2f}")
        print(f"  Resistance Level: ₹{trends['resistance_level']:.2f}")
    print()
    
    # Generate comprehensive report
    print("8. Generating comprehensive report...")
    report = system.generate_comprehensive_report()
    system.save_report(report)
    print()
    
    # Export portfolio summary
    print("9. Exporting portfolio summary...")
    system.export_portfolio_csv()
    print()
    
    # Show generated files
    print("=== Generated Files ===")
    if os.path.exists(system.data_directory):
        for file in os.listdir(system.data_directory):
            filepath = os.path.join(system.data_directory, file)
            size = os.path.getsize(filepath)
            print(f"  {file} ({size} bytes)")
    
    print("\n=== System Demonstration Complete ===")
    print("The integrated financial analysis system successfully:")
    print("✓ Loaded and processed market data")
    print("✓ Created and analyzed portfolio")
    print("✓ Assessed risk levels")
    print("✓ Generated comprehensive reports")
    print("✓ Exported data in multiple formats")

if __name__ == "__main__":
    main()
