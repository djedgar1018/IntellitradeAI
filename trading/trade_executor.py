"""
Trade Executor for IntelliTradeAI
Handles trade execution for stocks, options, and cryptocurrencies
Supports both paper trading and alert-based execution
"""

from typing import Dict, Optional, Any
from datetime import datetime
import uuid
from database.db_manager import DatabaseManager
from trading.mode_manager import TradingMode, TradingModeManager


class TradeExecutor:
    """
    Executes trades and manages orders
    Supports paper trading with simulated fills and alert-based execution
    """
    
    def __init__(self, db_manager: DatabaseManager, mode_manager: TradingModeManager):
        self.db = db_manager
        self.mode_manager = mode_manager
        self.paper_trading = True
        self.execution_history = []
    
    def execute_trade(self, trade_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a trade based on mode and parameters
        
        Args:
            trade_params: Dictionary with symbol, action, quantity, asset_type, etc.
            
        Returns:
            Execution result with trade ID and status
        """
        symbol = trade_params['symbol']
        action = trade_params['action']
        quantity = trade_params['quantity']
        asset_type = trade_params['asset_type']
        current_price = trade_params.get('current_price')
        
        signal_data = trade_params.get('signal', {})
        
        mode_decision = self.mode_manager.should_execute_trade(signal_data)
        
        if self.mode_manager.is_manual_mode():
            result = self._create_manual_trade_alert(trade_params, mode_decision)
        else:
            if mode_decision['should_execute']:
                result = self._execute_automatic_trade(trade_params, mode_decision)
            else:
                result = {
                    'success': False,
                    'reason': mode_decision['message'],
                    'mode': 'automatic',
                    'action': 'skipped'
                }
        
        self.execution_history.append({
            'timestamp': datetime.now(),
            'trade_params': trade_params,
            'result': result
        })
        
        return result
    
    def _execute_automatic_trade(self, trade_params: Dict, mode_decision: Dict) -> Dict[str, Any]:
        """Execute trade in automatic mode"""
        try:
            symbol = trade_params['symbol']
            action = trade_params['action']
            quantity = trade_params['quantity']
            asset_type = trade_params['asset_type']
            current_price = trade_params.get('current_price', 0)
            
            execution_params = mode_decision.get('execution_params', {})
            
            portfolio = self.db.get_portfolio()
            cash_balance = float(portfolio.get('cash_balance', 0))
            
            total_cost = current_price * quantity
            
            if action == 'BUY':
                if asset_type == 'option':
                    total_cost = current_price * quantity * 100
                
                if total_cost > cash_balance:
                    return {
                        'success': False,
                        'reason': f'Insufficient funds: ${cash_balance:.2f} available, ${total_cost:.2f} required',
                        'mode': 'automatic'
                    }
            
            trade_data = {
                'trade_id': str(uuid.uuid4()),
                'asset_type': asset_type,
                'symbol': symbol,
                'action': action,
                'quantity': quantity,
                'entry_price': current_price,
                'trading_mode': 'automatic',
                'status': 'open',
                'ai_confidence': trade_params.get('signal', {}).get('confidence'),
                'ai_signal': trade_params.get('signal', {}).get('action'),
                'notes': f'Auto-executed with {execution_params.get("stop_loss_percent", 0)}% SL, {execution_params.get("take_profit_percent", 0)}% TP',
                'fees': total_cost * 0.001
            }
            
            trade_id = self.db.log_trade(trade_data)
            
            if action == 'BUY':
                new_cash = cash_balance - total_cost - trade_data['fees']
                
                position_data = {
                    'position_id': str(uuid.uuid4()),
                    'asset_type': asset_type,
                    'symbol': symbol,
                    'quantity': quantity,
                    'avg_entry_price': current_price,
                    'current_price': current_price,
                    'total_invested': total_cost,
                    'current_value': total_cost,
                    'unrealized_pnl': 0,
                    'unrealized_pnl_percent': 0,
                    'status': 'active'
                }
                self.db.update_position(position_data)
            
            else:
                positions = self.db.get_active_positions()
                position = next((p for p in positions if p['symbol'] == symbol and p['asset_type'] == asset_type), None)
                
                if position:
                    avg_entry = float(position['avg_entry_price'])
                    pnl = (current_price - avg_entry) * quantity
                    new_cash = cash_balance + (current_price * quantity) - trade_data['fees']
                    
                    self.db.close_position(symbol, asset_type)
                else:
                    new_cash = cash_balance
            
            stop_loss_percent = execution_params.get('stop_loss_percent', 5.0)
            take_profit_percent = execution_params.get('take_profit_percent', 15.0)
            
            if action == 'BUY':
                stop_loss_price = current_price * (1 - stop_loss_percent / 100)
                take_profit_price = current_price * (1 + take_profit_percent / 100)
                
                self.db.create_alert({
                    'symbol': symbol,
                    'asset_type': asset_type,
                    'alert_type': 'stop_loss',
                    'target_price': stop_loss_price,
                    'current_price': current_price,
                    'action': 'SELL',
                    'quantity': quantity
                })
                
                self.db.create_alert({
                    'symbol': symbol,
                    'asset_type': asset_type,
                    'alert_type': 'take_profit',
                    'target_price': take_profit_price,
                    'current_price': current_price,
                    'action': 'SELL',
                    'quantity': quantity
                })
            
            portfolio_update = portfolio.copy()
            portfolio_update['cash_balance'] = new_cash
            portfolio_update['total_trades'] = int(portfolio.get('total_trades', 0)) + 1
            
            if asset_type == 'crypto':
                portfolio_update['crypto_balance'] = float(portfolio.get('crypto_balance', 0)) + total_cost if action == 'BUY' else float(portfolio.get('crypto_balance', 0)) - total_cost
            elif asset_type == 'stock':
                portfolio_update['stock_balance'] = float(portfolio.get('stock_balance', 0)) + total_cost if action == 'BUY' else float(portfolio.get('stock_balance', 0)) - total_cost
            elif asset_type == 'option':
                portfolio_update['options_balance'] = float(portfolio.get('options_balance', 0)) + total_cost if action == 'BUY' else float(portfolio.get('options_balance', 0)) - total_cost
            
            self.db.update_portfolio(portfolio_update)
            
            return {
                'success': True,
                'trade_id': trade_id,
                'mode': 'automatic',
                'action': 'executed',
                'symbol': symbol,
                'asset_type': asset_type,
                'quantity': quantity,
                'price': current_price,
                'total_cost': round(total_cost, 2),
                'fees': round(trade_data['fees'], 2),
                'new_cash_balance': round(new_cash, 2),
                'stop_loss_price': round(stop_loss_price, 2) if action == 'BUY' else None,
                'take_profit_price': round(take_profit_price, 2) if action == 'BUY' else None,
                'timestamp': datetime.now().isoformat(),
                'message': f'Automatically executed {action} of {quantity} {symbol} at ${current_price:.2f}'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'mode': 'automatic',
                'message': f'Trade execution failed: {str(e)}'
            }
    
    def _create_manual_trade_alert(self, trade_params: Dict, mode_decision: Dict) -> Dict[str, Any]:
        """Create alert/recommendation for manual mode"""
        try:
            symbol = trade_params['symbol']
            action = trade_params['action']
            quantity = trade_params['quantity']
            current_price = trade_params.get('current_price', 0)
            asset_type = trade_params['asset_type']
            
            alert_data = {
                'symbol': symbol,
                'asset_type': asset_type,
                'alert_type': 'price_target',
                'target_price': current_price,
                'current_price': current_price,
                'action': action,
                'quantity': quantity
            }
            
            alert_id = self.db.create_alert(alert_data)
            
            return {
                'success': True,
                'alert_id': alert_id,
                'mode': 'manual',
                'action': 'alert_created',
                'symbol': symbol,
                'recommendation': mode_decision.get('recommendation'),
                'signal': mode_decision.get('signal'),
                'message': f'Alert created for {action} {quantity} {symbol} at ${current_price:.2f}. Awaiting your confirmation.',
                'manual_instructions': {
                    'next_steps': 'Review the AI analysis and confirm to execute this trade',
                    'confirm_to_execute': True,
                    'estimated_cost': current_price * quantity if asset_type != 'option' else current_price * quantity * 100,
                    'ai_confidence': trade_params.get('signal', {}).get('confidence', 0)
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'mode': 'manual',
                'message': f'Failed to create manual trade alert: {str(e)}'
            }
    
    def confirm_manual_trade(self, alert_id: str) -> Dict[str, Any]:
        """User confirms and executes a manual trade"""
        try:
            alerts = self.db.get_active_alerts()
            alert = next((a for a in alerts if a['alert_id'] == alert_id), None)
            
            if not alert:
                return {
                    'success': False,
                    'message': 'Alert not found or already processed'
                }
            
            self.db.trigger_alert(alert_id)
            
            trade_params = {
                'symbol': alert['symbol'],
                'action': alert['action'],
                'quantity': float(alert['quantity']),
                'asset_type': alert['asset_type'],
                'current_price': float(alert['target_price']),
                'signal': {}
            }
            
            trade_data = {
                'trade_id': str(uuid.uuid4()),
                'asset_type': alert['asset_type'],
                'symbol': alert['symbol'],
                'action': alert['action'],
                'quantity': alert['quantity'],
                'entry_price': alert['target_price'],
                'trading_mode': 'manual',
                'status': 'open',
                'notes': f'Manually confirmed trade from alert {alert_id}',
                'fees': float(alert['target_price']) * float(alert['quantity']) * 0.001
            }
            
            trade_id = self.db.log_trade(trade_data)
            
            return {
                'success': True,
                'trade_id': trade_id,
                'alert_id': alert_id,
                'mode': 'manual',
                'action': 'executed',
                'message': f'Manual trade confirmed and executed for {alert["symbol"]}',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': f'Failed to confirm manual trade: {str(e)}'
            }
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of trade executions"""
        return {
            'total_executions': len(self.execution_history),
            'recent_executions': self.execution_history[-10:],
            'paper_trading_enabled': self.paper_trading,
            'current_mode': self.mode_manager.get_current_mode().value
        }
