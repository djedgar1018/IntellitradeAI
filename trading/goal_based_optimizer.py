"""
Goal-Based Strategy Optimizer for IntelliTradeAI
=================================================
Recommends optimized trading parameters based on:
- User's target goal (2x, 5x, 10x, custom)
- Timeframe (1 week, 1 month, 3 months)
- Asset class preference (stocks, crypto, forex, options)
- Risk tolerance (conservative, moderate, aggressive)

Includes precision, recall, F1 scoring for performance analysis.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from datetime import datetime, timedelta
import numpy as np


class RiskTolerance(Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    EXTREME = "extreme"


class AssetClass(Enum):
    STOCKS = "stocks"
    CRYPTO = "crypto"
    FOREX = "forex"
    OPTIONS = "options"
    ALL = "all"


@dataclass
class PerformanceMetrics:
    """Model performance metrics including precision, recall, F1"""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    true_positives: int = 0
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    
    def calculate_metrics(self):
        """Calculate precision, recall, F1 from confusion matrix"""
        total = self.true_positives + self.true_negatives + self.false_positives + self.false_negatives
        if total > 0:
            self.accuracy = (self.true_positives + self.true_negatives) / total
        
        precision_denom = self.true_positives + self.false_positives
        if precision_denom > 0:
            self.precision = self.true_positives / precision_denom
        
        recall_denom = self.true_positives + self.false_negatives
        if recall_denom > 0:
            self.recall = self.true_positives / recall_denom
        
        if self.precision + self.recall > 0:
            self.f1_score = 2 * (self.precision * self.recall) / (self.precision + self.recall)
        
        if self.total_trades > 0:
            self.win_rate = self.winning_trades / self.total_trades
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'accuracy': round(self.accuracy * 100, 2),
            'precision': round(self.precision * 100, 2),
            'recall': round(self.recall * 100, 2),
            'f1_score': round(self.f1_score * 100, 2),
            'win_rate': round(self.win_rate * 100, 2),
            'profit_factor': round(self.profit_factor, 2),
            'sharpe_ratio': round(self.sharpe_ratio, 2),
            'max_drawdown': round(self.max_drawdown, 2),
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'confusion_matrix': {
                'true_positives': self.true_positives,
                'true_negatives': self.true_negatives,
                'false_positives': self.false_positives,
                'false_negatives': self.false_negatives
            }
        }


@dataclass
class UserTradingPlan:
    """User's trading plan and goals"""
    target_return_percent: float  # e.g., 100 for 2x, 400 for 5x, 900 for 10x
    target_growth_multiple: float  # e.g., 2.0, 5.0, 10.0
    timeframe_days: int  # e.g., 7, 30, 90
    asset_class: AssetClass = AssetClass.ALL
    risk_tolerance: RiskTolerance = RiskTolerance.MODERATE
    starting_capital: float = 10000.0
    max_drawdown_limit: float = 30.0  # Maximum acceptable drawdown %
    preferred_hold_time_days: int = 2
    trades_per_day_limit: int = 20
    
    @classmethod
    def from_goal(cls, target_multiple: float, timeframe_days: int,
                  asset_class: str = 'all', risk_tolerance: str = 'moderate',
                  starting_capital: float = 10000.0) -> 'UserTradingPlan':
        """Create a trading plan from a simple goal"""
        return cls(
            target_return_percent=(target_multiple - 1) * 100,
            target_growth_multiple=target_multiple,
            timeframe_days=timeframe_days,
            asset_class=AssetClass(asset_class.lower()),
            risk_tolerance=RiskTolerance(risk_tolerance.lower()),
            starting_capital=starting_capital
        )


@dataclass
class OptimizedParameters:
    """Optimized trading parameters to meet user's goal"""
    base_risk_pct: float
    max_position_pct: float
    stop_loss_pct: float
    target_pct: float
    max_hold_days: int
    max_positions: int
    min_confidence: float
    pyramid_enabled: bool
    pyramid_max: int
    pyramid_add_pct: float
    win_streak_mult_max: float
    volatility_bonus_max: float
    trades_per_day_target: int
    required_win_rate: float
    required_avg_gain: float
    feasibility_score: float  # 0-100, how achievable is this goal
    warning_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'position_sizing': {
                'base_risk_pct': self.base_risk_pct,
                'max_position_pct': self.max_position_pct,
                'max_positions': self.max_positions
            },
            'exit_rules': {
                'stop_loss_pct': self.stop_loss_pct,
                'target_pct': self.target_pct,
                'max_hold_days': self.max_hold_days
            },
            'signal_filters': {
                'min_confidence': self.min_confidence
            },
            'compounding': {
                'pyramid_enabled': self.pyramid_enabled,
                'pyramid_max': self.pyramid_max,
                'pyramid_add_pct': self.pyramid_add_pct,
                'win_streak_mult_max': self.win_streak_mult_max,
                'volatility_bonus_max': self.volatility_bonus_max
            },
            'execution': {
                'trades_per_day_target': self.trades_per_day_target
            },
            'requirements': {
                'required_win_rate': round(self.required_win_rate * 100, 1),
                'required_avg_gain': round(self.required_avg_gain, 2)
            },
            'feasibility': {
                'score': round(self.feasibility_score, 1),
                'warning': self.warning_message
            }
        }


class GoalBasedOptimizer:
    """
    Optimizes trading parameters based on user's specific goals.
    Suggests the most realistic parameters to achieve target returns.
    """
    
    # Asset class characteristics
    ASSET_PROFILES = {
        'stocks': {
            'avg_daily_volatility': 2.5,
            'max_daily_volatility': 8.0,
            'typical_win_rate': 0.55,
            'typical_profit_factor': 1.8,
            'min_stop': 0.25,
            'max_stop': 2.0,
            'min_target': 3.0,
            'max_target': 15.0,
            'liquidity': 'high',
            'trading_hours': 6.5
        },
        'crypto': {
            'avg_daily_volatility': 5.0,
            'max_daily_volatility': 15.0,
            'typical_win_rate': 0.52,
            'typical_profit_factor': 1.6,
            'min_stop': 0.35,
            'max_stop': 3.0,
            'min_target': 5.0,
            'max_target': 25.0,
            'liquidity': 'high',
            'trading_hours': 24
        },
        'forex': {
            'avg_daily_volatility': 0.8,
            'max_daily_volatility': 3.0,
            'typical_win_rate': 0.58,
            'typical_profit_factor': 2.0,
            'min_stop': 0.10,
            'max_stop': 0.50,
            'min_target': 2.0,
            'max_target': 8.0,
            'liquidity': 'very_high',
            'trading_hours': 24
        },
        'options': {
            'avg_daily_volatility': 8.0,
            'max_daily_volatility': 30.0,
            'typical_win_rate': 0.48,
            'typical_profit_factor': 1.5,
            'min_stop': 0.15,
            'max_stop': 1.5,
            'min_target': 5.0,
            'max_target': 50.0,
            'liquidity': 'medium',
            'trading_hours': 6.5
        }
    }
    
    # Risk tolerance multipliers
    RISK_MULTIPLIERS = {
        RiskTolerance.CONSERVATIVE: 0.5,
        RiskTolerance.MODERATE: 1.0,
        RiskTolerance.AGGRESSIVE: 1.8,
        RiskTolerance.EXTREME: 3.0
    }
    
    def __init__(self):
        self.training_history: List[Dict[str, Any]] = []
        self.optimization_history: List[Dict[str, Any]] = []
    
    def calculate_required_performance(self, plan: UserTradingPlan) -> Dict[str, float]:
        """
        Calculate the required daily/weekly performance to meet the goal.
        Uses compound growth formula: Final = Initial * (1 + r)^n
        """
        target = plan.target_growth_multiple
        days = plan.timeframe_days
        
        # Calculate required daily return (compound)
        required_daily_return = (target ** (1 / days) - 1) * 100
        
        # Calculate required weekly return
        weeks = max(1, days / 7)
        required_weekly_return = (target ** (1 / weeks) - 1) * 100
        
        # Estimate required trades per day
        profile = self.ASSET_PROFILES.get(
            plan.asset_class.value if plan.asset_class != AssetClass.ALL else 'stocks',
            self.ASSET_PROFILES['stocks']
        )
        
        avg_gain_per_trade = profile['avg_daily_volatility'] * 0.6  # Capture 60% of volatility
        trades_per_day = max(1, required_daily_return / avg_gain_per_trade)
        
        return {
            'required_daily_return': required_daily_return,
            'required_weekly_return': required_weekly_return,
            'estimated_trades_per_day': trades_per_day,
            'days_remaining': days
        }
    
    def assess_feasibility(self, plan: UserTradingPlan) -> Tuple[float, str]:
        """
        Assess how feasible the user's goal is.
        Returns (feasibility_score 0-100, warning_message)
        """
        perf = self.calculate_required_performance(plan)
        daily_req = perf['required_daily_return']
        
        # Get asset profile
        asset = plan.asset_class.value if plan.asset_class != AssetClass.ALL else 'stocks'
        profile = self.ASSET_PROFILES.get(asset, self.ASSET_PROFILES['stocks'])
        
        # Compare required daily return to asset volatility
        volatility_ratio = daily_req / profile['avg_daily_volatility']
        
        # Score based on how realistic the goal is
        if volatility_ratio <= 0.3:
            score = 95
            warning = None
        elif volatility_ratio <= 0.6:
            score = 85
            warning = None
        elif volatility_ratio <= 1.0:
            score = 70
            warning = "Achievable with consistent execution"
        elif volatility_ratio <= 1.5:
            score = 55
            warning = "Requires favorable market conditions"
        elif volatility_ratio <= 2.0:
            score = 40
            warning = "Challenging - requires high win streaks"
        elif volatility_ratio <= 3.0:
            score = 25
            warning = "Very challenging - consider extending timeframe"
        else:
            score = 10
            warning = "Extremely ambitious - recommend lower target or longer timeframe"
        
        # Adjust for risk tolerance
        risk_mult = self.RISK_MULTIPLIERS[plan.risk_tolerance]
        if risk_mult >= 1.8 and volatility_ratio > 1.0:
            score = min(score + 15, 95)
        
        # Adjust for timeframe
        if plan.timeframe_days >= 90:
            score = min(score + 10, 95)
        elif plan.timeframe_days <= 7:
            score = max(score - 15, 5)
        
        return score, warning
    
    def optimize_for_goal(self, plan: UserTradingPlan) -> OptimizedParameters:
        """
        Generate optimized parameters to meet user's trading goal.
        """
        perf = self.calculate_required_performance(plan)
        feasibility, warning = self.assess_feasibility(plan)
        
        # Get asset profile
        asset = plan.asset_class.value if plan.asset_class != AssetClass.ALL else 'stocks'
        profile = self.ASSET_PROFILES.get(asset, self.ASSET_PROFILES['stocks'])
        
        # Risk multiplier
        risk_mult = self.RISK_MULTIPLIERS[plan.risk_tolerance]
        
        # Calculate required win rate and avg gain
        required_daily = perf['required_daily_return']
        trades_per_day = min(perf['estimated_trades_per_day'], plan.trades_per_day_limit)
        
        if trades_per_day > 0:
            required_avg_gain = required_daily / trades_per_day
        else:
            required_avg_gain = required_daily
        
        # Calculate required win rate given typical profit factor
        # WinRate * AvgWin - (1 - WinRate) * AvgLoss = RequiredReturn
        # Simplified: need higher win rate for lower gain targets
        typical_risk_reward = profile['max_target'] / profile['max_stop'] / 2
        required_win_rate = 0.5 + (0.1 * risk_mult)  # Base + risk adjustment
        required_win_rate = min(0.85, max(0.45, required_win_rate))
        
        # Position sizing based on goal aggressiveness
        goal_ratio = plan.target_growth_multiple / (plan.timeframe_days / 30)  # Normalize to monthly
        
        if goal_ratio <= 1.5:  # Conservative target
            base_risk = 12.0 * risk_mult
            max_position = 60.0 * risk_mult
        elif goal_ratio <= 3.0:  # Moderate target
            base_risk = 20.0 * risk_mult
            max_position = 75.0 * min(risk_mult, 1.2)
        elif goal_ratio <= 5.0:  # Aggressive target (5x monthly)
            base_risk = 28.0 * risk_mult
            max_position = 85.0 * min(risk_mult, 1.1)
        else:  # Extreme target (10x+ monthly)
            base_risk = 35.0 * risk_mult
            max_position = 92.0
        
        # Cap values
        base_risk = min(50.0, max(5.0, base_risk))
        max_position = min(95.0, max(30.0, max_position))
        
        # Stop loss based on asset and risk tolerance
        stop_base = (profile['min_stop'] + profile['max_stop']) / 2
        stop_loss = stop_base * (0.5 + 0.5 / risk_mult)  # Tighter stops for aggressive
        stop_loss = max(profile['min_stop'], min(profile['max_stop'], stop_loss))
        
        # Target based on risk:reward and goal
        target_ratio = 3.0 + (goal_ratio * 0.5)  # Higher goals need higher targets
        target = stop_loss * target_ratio
        target = max(profile['min_target'], min(profile['max_target'], target))
        
        # Confidence threshold (lower for more trades)
        if trades_per_day >= 10:
            min_conf = 40.0
        elif trades_per_day >= 5:
            min_conf = 50.0
        else:
            min_conf = 60.0
        
        # Pyramiding settings
        pyramid_enabled = goal_ratio >= 2.0
        pyramid_max = min(6, max(2, int(goal_ratio)))
        pyramid_add = min(80, 50 + goal_ratio * 10)
        
        # Win streak multiplier
        win_streak_mult = min(4.5, 1.5 + goal_ratio * 0.5)
        
        # Volatility bonus
        vol_bonus = min(1.5, 1.1 + goal_ratio * 0.08)
        
        # Max positions (fewer for concentration)
        if goal_ratio >= 5.0:
            max_positions = 5
        elif goal_ratio >= 3.0:
            max_positions = 7
        else:
            max_positions = 10
        
        return OptimizedParameters(
            base_risk_pct=round(base_risk, 1),
            max_position_pct=round(max_position, 1),
            stop_loss_pct=round(stop_loss, 2),
            target_pct=round(target, 1),
            max_hold_days=plan.preferred_hold_time_days,
            max_positions=max_positions,
            min_confidence=round(min_conf, 1),
            pyramid_enabled=pyramid_enabled,
            pyramid_max=pyramid_max,
            pyramid_add_pct=round(pyramid_add, 1),
            win_streak_mult_max=round(win_streak_mult, 2),
            volatility_bonus_max=round(vol_bonus, 2),
            trades_per_day_target=int(min(trades_per_day, plan.trades_per_day_limit)),
            required_win_rate=required_win_rate,
            required_avg_gain=required_avg_gain,
            feasibility_score=feasibility,
            warning_message=warning
        )
    
    def get_recommendation(self, 
                           target_multiple: float,
                           timeframe_days: int,
                           asset_class: str = 'all',
                           risk_tolerance: str = 'moderate',
                           starting_capital: float = 10000.0) -> Dict[str, Any]:
        """
        Get complete trading recommendation for user's goal.
        
        Args:
            target_multiple: Desired growth (e.g., 2.0 for 2x, 5.0 for 5x)
            timeframe_days: Days to achieve goal
            asset_class: 'stocks', 'crypto', 'forex', 'options', or 'all'
            risk_tolerance: 'conservative', 'moderate', 'aggressive', 'extreme'
            starting_capital: Starting balance
        
        Returns:
            Complete recommendation with parameters and analysis
        """
        plan = UserTradingPlan.from_goal(
            target_multiple=target_multiple,
            timeframe_days=timeframe_days,
            asset_class=asset_class,
            risk_tolerance=risk_tolerance,
            starting_capital=starting_capital
        )
        
        params = self.optimize_for_goal(plan)
        perf_requirements = self.calculate_required_performance(plan)
        
        # Calculate expected end balance
        expected_balance = starting_capital * target_multiple
        
        return {
            'user_goal': {
                'target_multiple': f'{target_multiple}x',
                'target_return': f'+{(target_multiple - 1) * 100:.0f}%',
                'timeframe_days': timeframe_days,
                'asset_class': asset_class,
                'risk_tolerance': risk_tolerance,
                'starting_capital': starting_capital,
                'target_balance': round(expected_balance, 2)
            },
            'performance_requirements': {
                'daily_return_needed': f'+{perf_requirements["required_daily_return"]:.2f}%',
                'weekly_return_needed': f'+{perf_requirements["required_weekly_return"]:.2f}%',
                'trades_per_day': perf_requirements['estimated_trades_per_day'],
                'required_win_rate': f'{params.required_win_rate * 100:.1f}%',
                'required_avg_gain_per_trade': f'{params.required_avg_gain:.2f}%'
            },
            'optimized_parameters': params.to_dict(),
            'feasibility_assessment': {
                'score': params.feasibility_score,
                'rating': self._get_feasibility_rating(params.feasibility_score),
                'message': params.warning_message or 'Goal appears achievable with consistent execution'
            },
            'recommendations': self._generate_recommendations(plan, params)
        }
    
    def _get_feasibility_rating(self, score: float) -> str:
        if score >= 80:
            return 'Highly Achievable'
        elif score >= 60:
            return 'Achievable'
        elif score >= 40:
            return 'Challenging'
        elif score >= 20:
            return 'Very Challenging'
        else:
            return 'Extremely Ambitious'
    
    def _generate_recommendations(self, plan: UserTradingPlan, params: OptimizedParameters) -> List[str]:
        """Generate actionable recommendations"""
        recs = []
        
        if params.feasibility_score < 40:
            recs.append(f"Consider extending timeframe to {plan.timeframe_days * 2} days for higher success probability")
            recs.append(f"Or reduce target to {plan.target_growth_multiple * 0.6:.1f}x for this timeframe")
        
        if plan.asset_class == AssetClass.ALL:
            recs.append("Focus on crypto and stocks for highest growth potential")
        
        if plan.risk_tolerance == RiskTolerance.CONSERVATIVE and plan.target_growth_multiple >= 3:
            recs.append("Aggressive targets require higher risk tolerance - consider 'aggressive' setting")
        
        if params.pyramid_enabled:
            recs.append(f"Pyramiding enabled - add {params.pyramid_add_pct:.0f}% to winners up to {params.pyramid_max}x")
        
        recs.append(f"Maintain win streaks for up to {params.win_streak_mult_max:.1f}x position sizing bonus")
        recs.append(f"Target high-volatility setups for up to {params.volatility_bonus_max:.0f}x bonus")
        
        return recs
    
    def compare_asset_classes(self, target_multiple: float, timeframe_days: int) -> Dict[str, Any]:
        """Compare feasibility across all asset classes for a given goal"""
        results = {}
        
        for asset in ['stocks', 'crypto', 'forex', 'options']:
            rec = self.get_recommendation(
                target_multiple=target_multiple,
                timeframe_days=timeframe_days,
                asset_class=asset,
                risk_tolerance='aggressive'
            )
            results[asset] = {
                'feasibility_score': rec['feasibility_assessment']['score'],
                'rating': rec['feasibility_assessment']['rating'],
                'required_win_rate': rec['performance_requirements']['required_win_rate'],
                'optimized_stop': rec['optimized_parameters']['exit_rules']['stop_loss_pct'],
                'optimized_target': rec['optimized_parameters']['exit_rules']['target_pct']
            }
        
        # Rank by feasibility
        ranked = sorted(results.items(), key=lambda x: x[1]['feasibility_score'], reverse=True)
        
        return {
            'goal': f'{target_multiple}x in {timeframe_days} days',
            'comparison': results,
            'recommended_order': [asset for asset, _ in ranked],
            'best_choice': ranked[0][0] if ranked else 'stocks'
        }


class ModelPerformanceTracker:
    """
    Tracks model performance with precision, recall, F1 metrics
    for each asset class and training round.
    """
    
    def __init__(self):
        self.training_rounds: Dict[str, List[PerformanceMetrics]] = {}
        self.asset_metrics: Dict[str, PerformanceMetrics] = {}
    
    def record_prediction(self, asset_class: str, predicted: str, actual: str, 
                          profit_loss: float = 0):
        """
        Record a prediction result for metric calculation.
        
        Args:
            asset_class: 'stocks', 'crypto', etc.
            predicted: 'BUY', 'SELL', 'HOLD'
            actual: What should have been predicted
            profit_loss: P&L if trade was taken
        """
        if asset_class not in self.asset_metrics:
            self.asset_metrics[asset_class] = PerformanceMetrics()
        
        metrics = self.asset_metrics[asset_class]
        
        # For BUY signals (positive class)
        if predicted == 'BUY':
            if actual == 'BUY':
                metrics.true_positives += 1
                if profit_loss > 0:
                    metrics.winning_trades += 1
                else:
                    metrics.losing_trades += 1
            else:
                metrics.false_positives += 1
                metrics.losing_trades += 1
        else:
            if actual == 'BUY':
                metrics.false_negatives += 1  # Missed opportunity
            else:
                metrics.true_negatives += 1
        
        metrics.total_trades += 1
        metrics.calculate_metrics()
    
    def get_metrics(self, asset_class: str) -> Dict[str, Any]:
        """Get performance metrics for an asset class"""
        if asset_class in self.asset_metrics:
            return self.asset_metrics[asset_class].to_dict()
        return PerformanceMetrics().to_dict()
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get metrics for all asset classes"""
        return {
            asset: metrics.to_dict()
            for asset, metrics in self.asset_metrics.items()
        }
    
    def start_training_round(self, round_name: str):
        """Start a new training round for tracking"""
        if round_name not in self.training_rounds:
            self.training_rounds[round_name] = []
    
    def end_training_round(self, round_name: str) -> Dict[str, Any]:
        """End training round and snapshot current metrics"""
        snapshot = {}
        for asset, metrics in self.asset_metrics.items():
            snapshot[asset] = metrics.to_dict()
            self.training_rounds.get(round_name, []).append(
                PerformanceMetrics(**{
                    'accuracy': metrics.accuracy,
                    'precision': metrics.precision,
                    'recall': metrics.recall,
                    'f1_score': metrics.f1_score,
                    'win_rate': metrics.win_rate,
                    'profit_factor': metrics.profit_factor
                })
            )
        return snapshot
    
    def get_training_history(self) -> Dict[str, List[Dict]]:
        """Get performance history across training rounds"""
        return {
            round_name: [m.to_dict() for m in metrics_list]
            for round_name, metrics_list in self.training_rounds.items()
        }


# Convenience functions
def get_optimized_params(target_multiple: float, timeframe_days: int,
                        asset_class: str = 'all', risk_tolerance: str = 'moderate') -> Dict[str, Any]:
    """Get optimized parameters for a trading goal"""
    optimizer = GoalBasedOptimizer()
    return optimizer.get_recommendation(
        target_multiple=target_multiple,
        timeframe_days=timeframe_days,
        asset_class=asset_class,
        risk_tolerance=risk_tolerance
    )


def compare_asset_feasibility(target_multiple: float, timeframe_days: int) -> Dict[str, Any]:
    """Compare goal feasibility across asset classes"""
    optimizer = GoalBasedOptimizer()
    return optimizer.compare_asset_classes(target_multiple, timeframe_days)
