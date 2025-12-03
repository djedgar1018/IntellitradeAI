"""
Options Chain Data Fetcher for IntelliTradeAI
Fetches real-time options data including Greeks, implied volatility, and open interest
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.stats import norm


class OptionsDataFetcher:
    """Fetches and processes options chain data for stocks"""
    
    def __init__(self):
        self.cache = {}
        self.cache_duration = 300
    
    def fetch_options_chain(self, symbol: str, expiration_date: Optional[str] = None) -> Dict:
        """
        Fetch complete options chain for a symbol
        
        Args:
            symbol: Stock ticker symbol
            expiration_date: Optional specific expiration date (YYYY-MM-DD)
            
        Returns:
            Dictionary containing calls and puts dataframes with all Greeks
        """
        try:
            ticker = yf.Ticker(symbol)
            
            available_expirations = ticker.options
            if not available_expirations:
                return {'error': f'No options available for {symbol}'}
            
            if expiration_date and expiration_date not in available_expirations:
                return {'error': f'Expiration date {expiration_date} not available'}
            
            target_expiration = expiration_date if expiration_date else available_expirations[0]
            
            options_chain = ticker.option_chain(target_expiration)
            calls = options_chain.calls
            puts = options_chain.puts
            
            current_price = self._get_current_price(ticker)
            
            calls_processed = self._process_options_data(
                calls, current_price, target_expiration, 'CALL'
            )
            puts_processed = self._process_options_data(
                puts, current_price, target_expiration, 'PUT'
            )
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'expiration_date': target_expiration,
                'calls': calls_processed,
                'puts': puts_processed,
                'available_expirations': list(available_expirations),
                'fetched_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {'error': f'Failed to fetch options for {symbol}: {str(e)}'}
    
    def _process_options_data(self, df: pd.DataFrame, current_price: float, 
                               expiration: str, option_type: str) -> pd.DataFrame:
        """Process options data and add calculated fields"""
        if df.empty:
            return df
        
        df = df.copy()
        
        df['inTheMoney'] = df['strike'] < current_price if option_type == 'CALL' else df['strike'] > current_price
        
        df['midPrice'] = (df['bid'] + df['ask']) / 2
        
        try:
            expiration_dt = datetime.strptime(expiration, '%Y-%m-%d')
            days_to_expiry = (expiration_dt - datetime.now()).days
            df['daysToExpiry'] = days_to_expiry
            
            if 'impliedVolatility' not in df.columns:
                df['impliedVolatility'] = 0
            
            if days_to_expiry > 0:
                df = self._calculate_greeks_bulk(df, current_price, days_to_expiry, option_type)
        except Exception as e:
            print(f"Error calculating Greeks: {e}")
        
        df['percentChange'] = ((df['lastPrice'] - df['strike']) / df['strike'] * 100).round(2)
        df['breakeven'] = df['strike'] + df['lastPrice'] if option_type == 'CALL' else df['strike'] - df['lastPrice']
        
        df = df.sort_values('strike')
        
        return df
    
    def _calculate_greeks_bulk(self, df: pd.DataFrame, spot_price: float, 
                                days_to_expiry: int, option_type: str) -> pd.DataFrame:
        """Calculate Greeks for all options in bulk using Black-Scholes"""
        df = df.copy()
        
        risk_free_rate = 0.045
        
        time_to_expiry = days_to_expiry / 365.0
        
        if time_to_expiry <= 0:
            time_to_expiry = 0.001
        
        for idx, row in df.iterrows():
            try:
                strike = row['strike']
                sigma = row.get('impliedVolatility', 0.3)
                
                if sigma <= 0 or pd.isna(sigma):
                    sigma = 0.3
                
                greeks = self._black_scholes_greeks(
                    spot_price, strike, time_to_expiry, risk_free_rate, sigma, option_type
                )
                
                df.loc[idx, 'delta'] = greeks['delta']
                df.loc[idx, 'gamma'] = greeks['gamma']
                df.loc[idx, 'theta'] = greeks['theta']
                df.loc[idx, 'vega'] = greeks['vega']
                df.loc[idx, 'rho'] = greeks['rho']
                
            except Exception as e:
                df.loc[idx, 'delta'] = 0
                df.loc[idx, 'gamma'] = 0
                df.loc[idx, 'theta'] = 0
                df.loc[idx, 'vega'] = 0
                df.loc[idx, 'rho'] = 0
        
        return df
    
    def _black_scholes_greeks(self, S: float, K: float, T: float, r: float, 
                               sigma: float, option_type: str) -> Dict[str, float]:
        """
        Calculate Greeks using Black-Scholes formula
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Implied volatility
            option_type: 'CALL' or 'PUT'
        """
        try:
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            if option_type == 'CALL':
                delta = norm.cdf(d1)
                theta_factor = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
                rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
            else:
                delta = -norm.cdf(-d1)
                theta_factor = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)
                rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
            
            gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
            theta = theta_factor / 365
            vega = S * norm.pdf(d1) * np.sqrt(T) / 100
            
            return {
                'delta': round(delta, 4),
                'gamma': round(gamma, 6),
                'theta': round(theta, 4),
                'vega': round(vega, 4),
                'rho': round(rho, 4)
            }
        except Exception as e:
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
    
    def _get_current_price(self, ticker) -> float:
        """Get current stock price"""
        try:
            info = ticker.info
            return info.get('regularMarketPrice') or info.get('currentPrice') or 0
        except:
            try:
                hist = ticker.history(period='1d')
                if not hist.empty:
                    return hist['Close'].iloc[-1]
            except:
                pass
        return 0
    
    def find_optimal_options(self, symbol: str, strategy: str = 'aggressive') -> Dict:
        """
        Find optimal options based on AI analysis
        
        Args:
            symbol: Stock ticker
            strategy: 'conservative', 'moderate', or 'aggressive'
            
        Returns:
            Recommended options with rationale
        """
        chain_data = self.fetch_options_chain(symbol)
        
        if 'error' in chain_data:
            return chain_data
        
        calls = chain_data['calls']
        puts = chain_data['puts']
        current_price = chain_data['current_price']
        
        if calls.empty and puts.empty:
            return {'error': 'No options data available'}
        
        recommendations = {
            'symbol': symbol,
            'current_price': current_price,
            'strategy': strategy,
            'call_recommendations': [],
            'put_recommendations': []
        }
        
        if not calls.empty:
            if strategy == 'conservative':
                itm_calls = calls[calls['inTheMoney'] == True]
                if not itm_calls.empty:
                    best_call = itm_calls.iloc[0]
                    recommendations['call_recommendations'].append({
                        'strike': float(best_call['strike']),
                        'premium': float(best_call['lastPrice']),
                        'delta': float(best_call.get('delta', 0)),
                        'breakeven': float(best_call.get('breakeven', 0)),
                        'reason': 'In-the-money for conservative capital preservation'
                    })
            
            elif strategy == 'aggressive':
                otm_calls = calls[calls['inTheMoney'] == False]
                if not otm_calls.empty:
                    high_volume = otm_calls.nlargest(3, 'volume')
                    for idx, call in high_volume.iterrows():
                        recommendations['call_recommendations'].append({
                            'strike': float(call['strike']),
                            'premium': float(call['lastPrice']),
                            'delta': float(call.get('delta', 0)),
                            'volume': int(call['volume']),
                            'breakeven': float(call.get('breakeven', 0)),
                            'reason': 'Out-of-money with high volume for leveraged upside'
                        })
            
            else:
                atm_calls = calls.iloc[(calls['strike'] - current_price).abs().argsort()[:3]]
                for idx, call in atm_calls.iterrows():
                    recommendations['call_recommendations'].append({
                        'strike': float(call['strike']),
                        'premium': float(call['lastPrice']),
                        'delta': float(call.get('delta', 0)),
                        'breakeven': float(call.get('breakeven', 0)),
                        'reason': 'At-the-money for balanced risk-reward'
                    })
        
        if not puts.empty:
            if strategy == 'conservative':
                itm_puts = puts[puts['inTheMoney'] == True]
                if not itm_puts.empty:
                    best_put = itm_puts.iloc[0]
                    recommendations['put_recommendations'].append({
                        'strike': float(best_put['strike']),
                        'premium': float(best_put['lastPrice']),
                        'delta': float(best_put.get('delta', 0)),
                        'breakeven': float(best_put.get('breakeven', 0)),
                        'reason': 'In-the-money protective put'
                    })
            
            elif strategy == 'aggressive':
                otm_puts = puts[puts['inTheMoney'] == False]
                if not otm_puts.empty:
                    high_volume = otm_puts.nlargest(3, 'volume')
                    for idx, put in high_volume.iterrows():
                        recommendations['put_recommendations'].append({
                            'strike': float(put['strike']),
                            'premium': float(put['lastPrice']),
                            'delta': float(put.get('delta', 0)),
                            'volume': int(put['volume']),
                            'breakeven': float(put.get('breakeven', 0)),
                            'reason': 'Out-of-money put for speculative downside'
                        })
        
        return recommendations
    
    def analyze_option_strategy(self, symbol: str, option_type: str, strike: float, 
                                 quantity: int, expiration: str) -> Dict:
        """
        Analyze a specific option strategy
        
        Returns:
            Risk analysis, max profit/loss, breakeven, probability of profit
        """
        chain_data = self.fetch_options_chain(symbol, expiration)
        
        if 'error' in chain_data:
            return chain_data
        
        current_price = chain_data['current_price']
        options_df = chain_data['calls'] if option_type == 'CALL' else chain_data['puts']
        
        option_row = options_df[options_df['strike'] == strike]
        
        if option_row.empty:
            return {'error': f'No {option_type} option found at strike {strike}'}
        
        option = option_row.iloc[0]
        premium = float(option['lastPrice'])
        total_cost = premium * quantity * 100
        
        breakeven = strike + premium if option_type == 'CALL' else strike - premium
        
        if option_type == 'CALL':
            max_loss = total_cost
            max_profit_unlimited = True
            max_profit = 'Unlimited'
            profit_at_20_percent = ((current_price * 1.2) - strike - premium) * quantity * 100
        else:
            max_loss = total_cost
            max_profit = (strike - premium) * quantity * 100
            max_profit_unlimited = False
            profit_at_20_percent = (strike - (current_price * 0.8) - premium) * quantity * 100
        
        distance_to_breakeven = ((breakeven - current_price) / current_price) * 100
        
        return {
            'symbol': symbol,
            'option_type': option_type,
            'strike': strike,
            'current_price': current_price,
            'premium': premium,
            'quantity': quantity,
            'total_cost': round(total_cost, 2),
            'breakeven': round(breakeven, 2),
            'distance_to_breakeven_percent': round(distance_to_breakeven, 2),
            'max_loss': round(max_loss, 2),
            'max_profit': max_profit,
            'estimated_profit_20pct_move': round(profit_at_20_percent, 2) if profit_at_20_percent else 0,
            'delta': float(option.get('delta', 0)),
            'theta': float(option.get('theta', 0)),
            'implied_volatility': float(option.get('impliedVolatility', 0)),
            'days_to_expiry': int(option.get('daysToExpiry', 0)),
            'risk_reward_ratio': 'Unlimited' if max_profit_unlimited else round(float(max_profit) / max_loss, 2) if max_loss > 0 else 0,
            'recommendation': self._generate_option_recommendation(
                option_type, distance_to_breakeven, float(option.get('delta', 0)), 
                float(option.get('impliedVolatility', 0))
            )
        }
    
    def _generate_option_recommendation(self, option_type: str, distance_to_breakeven: float,
                                         delta: float, iv: float) -> str:
        """Generate AI recommendation for an option"""
        if abs(distance_to_breakeven) < 5:
            confidence = "HIGH"
        elif abs(distance_to_breakeven) < 10:
            confidence = "MODERATE"
        else:
            confidence = "LOW"
        
        if option_type == 'CALL':
            if delta > 0.7:
                return f"{confidence} confidence - Deep ITM call with high delta, behaves like stock"
            elif delta > 0.5:
                return f"{confidence} confidence - ATM call, balanced risk-reward"
            else:
                return f"{confidence} confidence - OTM call, high risk but high potential return"
        else:
            if abs(delta) > 0.7:
                return f"{confidence} confidence - Deep ITM put, strong downside protection"
            elif abs(delta) > 0.5:
                return f"{confidence} confidence - ATM put, balanced hedge"
            else:
                return f"{confidence} confidence - OTM put, cheap insurance"
