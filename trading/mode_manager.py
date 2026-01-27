"""
Trading Mode Manager for IntelliTradeAI
Handles Manual and Automatic trading modes with different execution logic
V22 Optimized for 5x/10x Growth Targets
"""

from enum import Enum
from typing import Dict, Optional, Any
from datetime import datetime

try:
    from trading.v22_scalp_config import V22ScalpConfig
    V22_AVAILABLE = True
except ImportError:
    V22_AVAILABLE = False

try:
    from trading.goal_based_optimizer import (
        GoalBasedOptimizer, UserTradingPlan, 
        ModelPerformanceTracker, RiskTolerance, AssetClass
    )
    OPTIMIZER_AVAILABLE = True
except ImportError:
    OPTIMIZER_AVAILABLE = False


class TradingMode(Enum):
    """Trading mode enumeration"""
    MANUAL = "manual"
    AUTOMATIC = "automatic"


class TradingModeManager:
    """
    Manages trading mode switching and mode-specific behavior
    
    Manual Mode: AI provides recommendations, user makes final decision
    Automatic Mode: AI executes trades autonomously based on signals
    
    V22 Integration: Uses optimized scalping parameters for 5x/10x growth
    """
    
    def __init__(self, initial_mode: TradingMode = TradingMode.MANUAL):
        self.current_mode = initial_mode
        self.mode_history = [(datetime.now(), initial_mode)]
        self.strategy_version = 22
        
        # V22 Optimized Auto Trading Config
        self.auto_trade_config = {
            'enabled': False,
            'strategy_version': 22,
            'min_confidence': 45.0,  # V22: Lower threshold for more trades
            'max_position_size_percent': 85.0,  # V22: Aggressive positioning
            'stop_loss_percent': 0.35,  # V22: Tight scalping stops
            'take_profit_percent': 9.5,  # V22: Quick profit targets
            'max_daily_trades': 50,  # V22: High frequency scalping
            'max_loss_per_day': 1500.0,  # V22: Higher risk tolerance
            'allowed_asset_types': ['stock', 'crypto', 'option', 'forex'],
            'enabled_assets': [],
            # V22 Scalping additions
            'max_hold_days': 2,
            'pyramid_enabled': True,
            'pyramid_max': 5,
            'pyramid_add_percent': 75.0,
            'win_streak_multiplier_enabled': True,
            'win_streak_multiplier_max': 3.8,
            'volatility_bonus_enabled': True,
            'volatility_bonus_max': 1.35
        }
        self.manual_trade_config = {
            'require_confirmation': True,
            'show_ai_analysis': True,
            'show_risk_metrics': True,
            'alert_on_signal': True
        }
        self.asset_modes = {
            'crypto': 'manual',
            'stocks': 'manual',
            'options': 'manual',
            'forex': 'manual'
        }
        # Track win streaks for position sizing
        self.win_streak = 0
        self.consecutive_losses = 0
        self.best_streak = 0
    
    def get_v22_config(self, asset_class: str = 'stocks') -> Dict[str, Any]:
        """Get V22 asset-specific configuration"""
        if V22_AVAILABLE:
            config = V22ScalpConfig.get_config(asset_class)
            return {
                'max_positions': config.max_positions,
                'base_risk_pct': config.base_risk_pct,
                'max_position_pct': config.max_position_pct,
                'stop_loss_pct': config.stop_loss_pct,
                'target_pct': config.target_pct,
                'max_hold_days': config.max_hold_days
            }
        # Fallback defaults
        return {
            'max_positions': 8,
            'base_risk_pct': 28.0,
            'max_position_pct': 82.0,
            'stop_loss_pct': 0.35,
            'target_pct': 9.5,
            'max_hold_days': 2
        }
    
    def calculate_position_multiplier(self, signal_strength: int, atr_pct: float = 0) -> float:
        """Calculate position size multiplier based on V22 rules"""
        if V22_AVAILABLE:
            return V22ScalpConfig.calculate_position_multiplier(
                signal_strength, self.win_streak, atr_pct, self.consecutive_losses
            )
        # Fallback calculation
        mult = 1.0
        if signal_strength >= 80:
            mult = 4.8
        elif signal_strength >= 60:
            mult = 3.4
        elif signal_strength >= 40:
            mult = 2.3
        
        if self.win_streak >= 10:
            mult *= 2.8
        elif self.win_streak >= 6:
            mult *= 2.0
        elif self.win_streak >= 3:
            mult *= 1.32
        
        return mult
    
    def record_trade_result(self, won: bool):
        """Record trade result for win streak tracking"""
        if won:
            self.win_streak += 1
            self.consecutive_losses = 0
            if self.win_streak > self.best_streak:
                self.best_streak = self.win_streak
        else:
            self.win_streak = 0
            self.consecutive_losses += 1
    
    def get_optimized_params_for_goal(self, 
                                       target_multiple: float,
                                       timeframe_days: int,
                                       asset_class: str = 'all',
                                       risk_tolerance: str = 'moderate',
                                       starting_capital: float = 10000.0) -> Dict[str, Any]:
        """
        Get optimized trading parameters for user's specific goal.
        
        Args:
            target_multiple: Desired growth (2.0 for 2x, 5.0 for 5x, etc.)
            timeframe_days: Days to achieve goal
            asset_class: 'stocks', 'crypto', 'forex', 'options', or 'all'
            risk_tolerance: 'conservative', 'moderate', 'aggressive', 'extreme'
            starting_capital: Starting balance
        
        Returns:
            Complete recommendation with optimized parameters
        """
        if OPTIMIZER_AVAILABLE:
            optimizer = GoalBasedOptimizer()
            return optimizer.get_recommendation(
                target_multiple=target_multiple,
                timeframe_days=timeframe_days,
                asset_class=asset_class,
                risk_tolerance=risk_tolerance,
                starting_capital=starting_capital
            )
        
        # Fallback to V22 defaults with clear warning
        return {
            'user_goal': {
                'target_multiple': f'{target_multiple}x',
                'timeframe_days': timeframe_days,
                'asset_class': asset_class,
                'starting_capital': starting_capital
            },
            'optimized_parameters': {
                'position_sizing': {
                    'base_risk_pct': 28.0,
                    'max_position_pct': 85.0,
                    'max_positions': 8
                },
                'exit_rules': {
                    'stop_loss_pct': 0.35,
                    'target_pct': 9.5,
                    'max_hold_days': 2
                },
                'signal_filters': {
                    'min_confidence': 45.0
                },
                'compounding': {
                    'pyramid_enabled': True,
                    'pyramid_max': 5,
                    'pyramid_add_pct': 75.0,
                    'win_streak_mult_max': 3.8,
                    'volatility_bonus_max': 1.35
                },
                'execution': {
                    'trades_per_day_target': 5
                }
            },
            'performance_requirements': {
                'daily_return_needed': 'N/A - Optimizer unavailable',
                'weekly_return_needed': 'N/A - Optimizer unavailable',
                'required_win_rate': '55%',
                'required_avg_gain_per_trade': '2.5%'
            },
            'feasibility_assessment': {
                'score': 0,
                'rating': 'Unknown - Optimizer Unavailable',
                'message': 'Goal-based optimizer not available. Using V22 default parameters which may not align with your specific goal.'
            },
            'recommendations': [
                'Install goal_based_optimizer module for customized recommendations',
                'V22 defaults are optimized for 5x monthly returns',
                'Consider adjusting parameters manually if your goal differs'
            ]
        }
    
    def apply_goal_parameters(self, recommendation: Dict[str, Any]) -> bool:
        """Apply all optimized parameters from goal-based recommendation"""
        try:
            params = recommendation.get('optimized_parameters', {})
            
            if 'position_sizing' in params:
                sizing = params['position_sizing']
                self.auto_trade_config['max_position_size_percent'] = sizing.get('max_position_pct', 85.0)
                self.auto_trade_config['base_risk_percent'] = sizing.get('base_risk_pct', 28.0)
                self.auto_trade_config['max_positions'] = sizing.get('max_positions', 8)
            
            if 'exit_rules' in params:
                exits = params['exit_rules']
                self.auto_trade_config['stop_loss_percent'] = exits.get('stop_loss_pct', 0.35)
                self.auto_trade_config['take_profit_percent'] = exits.get('target_pct', 9.5)
                self.auto_trade_config['max_hold_days'] = exits.get('max_hold_days', 2)
            
            if 'signal_filters' in params:
                self.auto_trade_config['min_confidence'] = params['signal_filters'].get('min_confidence', 45.0)
            
            if 'compounding' in params:
                comp = params['compounding']
                self.auto_trade_config['pyramid_enabled'] = comp.get('pyramid_enabled', True)
                self.auto_trade_config['pyramid_max'] = comp.get('pyramid_max', 5)
                self.auto_trade_config['pyramid_add_percent'] = comp.get('pyramid_add_pct', 75.0)
                self.auto_trade_config['win_streak_multiplier_max'] = comp.get('win_streak_mult_max', 3.8)
                self.auto_trade_config['volatility_bonus_max'] = comp.get('volatility_bonus_max', 1.35)
            
            if 'execution' in params:
                self.auto_trade_config['trades_per_day_target'] = params['execution'].get('trades_per_day_target', 5)
            
            if 'requirements' in params:
                req = params['requirements']
                self.auto_trade_config['required_win_rate'] = req.get('required_win_rate', 55.0)
                self.auto_trade_config['required_avg_gain'] = req.get('required_avg_gain', 2.0)
            
            return True
        except Exception as e:
            print(f"Error applying goal parameters: {e}")
            return False
    
    def compare_asset_feasibility(self, target_multiple: float, timeframe_days: int) -> Dict[str, Any]:
        """Compare goal feasibility across asset classes"""
        if OPTIMIZER_AVAILABLE:
            optimizer = GoalBasedOptimizer()
            return optimizer.compare_asset_classes(target_multiple, timeframe_days)
        
        return {
            'goal': f'{target_multiple}x in {timeframe_days} days',
            'recommended_order': ['crypto', 'stocks', 'options', 'forex'],
            'best_choice': 'crypto'
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
    
    def set_asset_mode(self, asset_type: str, mode: str) -> Dict[str, Any]:
        """
        Set trading mode for a specific asset type
        
        Args:
            asset_type: 'crypto', 'stocks', or 'options'
            mode: 'manual' or 'automatic'
            
        Returns:
            Confirmation with updated configuration
        """
        if asset_type not in self.asset_modes:
            return {
                'success': False,
                'error': f'Unknown asset type: {asset_type}'
            }
        
        if mode not in ['manual', 'automatic']:
            return {
                'success': False,
                'error': f'Invalid mode: {mode}. Must be "manual" or "automatic"'
            }
        
        old_mode = self.asset_modes[asset_type]
        self.asset_modes[asset_type] = mode
        
        if mode == 'automatic':
            if asset_type not in self.auto_trade_config['enabled_assets']:
                self.auto_trade_config['enabled_assets'].append(asset_type)
        else:
            if asset_type in self.auto_trade_config['enabled_assets']:
                self.auto_trade_config['enabled_assets'].remove(asset_type)
        
        self.auto_trade_config['enabled'] = len(self.auto_trade_config['enabled_assets']) > 0
        
        return {
            'success': True,
            'asset_type': asset_type,
            'previous_mode': old_mode,
            'current_mode': mode,
            'enabled_assets': self.auto_trade_config['enabled_assets'],
            'auto_trading_enabled': self.auto_trade_config['enabled'],
            'timestamp': datetime.now().isoformat()
        }
    
    def get_asset_mode(self, asset_type: str) -> str:
        """Get current trading mode for specific asset type"""
        return self.asset_modes.get(asset_type, 'manual')
    
    def is_asset_auto_enabled(self, asset_type: str) -> bool:
        """Check if automatic trading is enabled for specific asset type"""
        return self.asset_modes.get(asset_type, 'manual') == 'automatic'
    
    def get_all_asset_modes(self) -> Dict[str, str]:
        """Get trading modes for all asset types"""
        return self.asset_modes.copy()
