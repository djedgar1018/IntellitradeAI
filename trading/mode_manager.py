"""
Trading Mode Manager for IntelliTradeAI
Handles Manual and Automatic trading modes with different execution logic
"""

from enum import Enum
from typing import Dict, Optional, Any
from datetime import datetime


class TradingMode(Enum):
    """Trading mode enumeration"""
    MANUAL = "manual"
    AUTOMATIC = "automatic"


class TradingModeManager:
    """
    Manages trading mode switching and mode-specific behavior
    
    Manual Mode: AI provides recommendations, user makes final decision
    Automatic Mode: AI executes trades autonomously based on signals
    """
    
    def __init__(self, initial_mode: TradingMode = TradingMode.MANUAL):
        self.current_mode = initial_mode
        self.mode_history = [(datetime.now(), initial_mode)]
        self.auto_trade_config = {
            'enabled': False,
            'min_confidence': 70.0,
            'max_position_size_percent': 10.0,
            'stop_loss_percent': 5.0,
            'take_profit_percent': 15.0,
            'max_daily_trades': 10,
            'max_loss_per_day': 500.0,
            'allowed_asset_types': ['stock', 'crypto', 'option']
        }
        self.manual_trade_config = {
            'require_confirmation': True,
            'show_ai_analysis': True,
            'show_risk_metrics': True,
            'alert_on_signal': True
        }
    
    def switch_mode(self, new_mode: TradingMode) -> Dict[str, Any]:
        """
        Switch between manual and automatic modes
        
        Args:
            new_mode: Target trading mode
            
        Returns:
            Confirmation with mode details
        """
        old_mode = self.current_mode
        self.current_mode = new_mode
        self.mode_history.append((datetime.now(), new_mode))
        
        if new_mode == TradingMode.AUTOMATIC:
            self.auto_trade_config['enabled'] = True
        else:
            self.auto_trade_config['enabled'] = False
        
        return {
            'success': True,
            'previous_mode': old_mode.value,
            'current_mode': new_mode.value,
            'timestamp': datetime.now().isoformat(),
            'message': f'Switched from {old_mode.value} to {new_mode.value} mode',
            'active_config': self.get_active_config()
        }
    
    def get_current_mode(self) -> TradingMode:
        """Get current trading mode"""
        return self.current_mode
    
    def is_manual_mode(self) -> bool:
        """Check if currently in manual mode"""
        return self.current_mode == TradingMode.MANUAL
    
    def is_automatic_mode(self) -> bool:
        """Check if currently in automatic mode"""
        return self.current_mode == TradingMode.AUTOMATIC
    
    def get_active_config(self) -> Dict[str, Any]:
        """Get configuration for current mode"""
        if self.is_automatic_mode():
            return {
                'mode': 'automatic',
                'config': self.auto_trade_config,
                'description': 'AI will execute trades automatically based on signals and confidence thresholds'
            }
        else:
            return {
                'mode': 'manual',
                'config': self.manual_trade_config,
                'description': 'AI provides recommendations as assists, user makes final trading decisions'
            }
    
    def update_auto_config(self, config_updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update automatic trading configuration"""
        for key, value in config_updates.items():
            if key in self.auto_trade_config:
                self.auto_trade_config[key] = value
        
        return {
            'success': True,
            'updated_config': self.auto_trade_config,
            'message': 'Automatic trading configuration updated'
        }
    
    def update_manual_config(self, config_updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update manual trading configuration"""
        for key, value in config_updates.items():
            if key in self.manual_trade_config:
                self.manual_trade_config[key] = value
        
        return {
            'success': True,
            'updated_config': self.manual_trade_config,
            'message': 'Manual trading configuration updated'
        }
    
    def should_execute_trade(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determine if a trade should be executed based on current mode and signal
        
        Args:
            signal: Trading signal with confidence, action, etc.
            
        Returns:
            Decision with rationale
        """
        if self.is_manual_mode():
            return {
                'should_execute': False,
                'mode': 'manual',
                'action': 'await_user_confirmation',
                'message': 'Manual mode: Present signal to user for approval',
                'signal': signal,
                'recommendation': self._generate_manual_recommendation(signal)
            }
        
        else:
            confidence = signal.get('confidence', 0)
            asset_type = signal.get('asset_type', '')
            
            can_execute = self._validate_auto_execution(signal, confidence, asset_type)
            
            if can_execute['valid']:
                return {
                    'should_execute': True,
                    'mode': 'automatic',
                    'action': 'execute_immediately',
                    'message': 'Automatic mode: Executing trade based on AI signal',
                    'signal': signal,
                    'execution_params': can_execute['params']
                }
            else:
                return {
                    'should_execute': False,
                    'mode': 'automatic',
                    'action': 'skip_trade',
                    'message': f'Auto-trade skipped: {can_execute["reason"]}',
                    'signal': signal
                }
    
    def _validate_auto_execution(self, signal: Dict, confidence: float, 
                                  asset_type: str) -> Dict[str, Any]:
        """Validate if signal meets automatic execution criteria"""
        config = self.auto_trade_config
        
        if not config['enabled']:
            return {'valid': False, 'reason': 'Automatic trading disabled'}
        
        if confidence < config['min_confidence']:
            return {
                'valid': False,
                'reason': f'Confidence {confidence}% below threshold {config["min_confidence"]}%'
            }
        
        if asset_type not in config['allowed_asset_types']:
            return {
                'valid': False,
                'reason': f'Asset type {asset_type} not allowed in auto-trade config'
            }
        
        return {
            'valid': True,
            'reason': 'Signal meets automatic execution criteria',
            'params': {
                'stop_loss_percent': config['stop_loss_percent'],
                'take_profit_percent': config['take_profit_percent'],
                'max_position_size_percent': config['max_position_size_percent']
            }
        }
    
    def _generate_manual_recommendation(self, signal: Dict) -> str:
        """Generate recommendation text for manual mode"""
        action = signal.get('action', 'HOLD')
        confidence = signal.get('confidence', 0)
        symbol = signal.get('symbol', 'Unknown')
        
        if confidence >= 75:
            strength = "STRONG"
        elif confidence >= 60:
            strength = "MODERATE"
        else:
            strength = "WEAK"
        
        return (
            f"{strength} {action} signal for {symbol} with {confidence}% confidence. "
            f"AI recommends {action} but awaits your final decision. "
            f"Review the analysis and confirm to execute."
        )
    
    def get_mode_stats(self) -> Dict[str, Any]:
        """Get statistics about mode usage"""
        total_switches = len(self.mode_history) - 1
        
        manual_count = sum(1 for _, mode in self.mode_history if mode == TradingMode.MANUAL)
        auto_count = sum(1 for _, mode in self.mode_history if mode == TradingMode.AUTOMATIC)
        
        return {
            'current_mode': self.current_mode.value,
            'total_mode_switches': total_switches,
            'manual_activations': manual_count,
            'automatic_activations': auto_count,
            'mode_history': [
                {'timestamp': ts.isoformat(), 'mode': mode.value}
                for ts, mode in self.mode_history[-10:]
            ]
        }
