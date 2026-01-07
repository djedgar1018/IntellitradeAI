"""Database persistence for Discord trade analysis."""

import os
import json
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime
from typing import List, Dict, Optional, Any

from discord_integration.trade_parser import ParsedTrade
from discord_integration.trade_analyzer import TraderProfile, TradingPattern


class DiscordDBPersistence:
    """Handles database persistence for Discord trade analysis."""
    
    def __init__(self):
        self.db_url = os.getenv('DATABASE_URL')
        self.demo_mode = not bool(self.db_url)
        if self.demo_mode:
            print("WARNING: DATABASE_URL not set. Discord integration running in demo mode.")
    
    def get_connection(self):
        """Get database connection."""
        if self.demo_mode:
            return None
        return psycopg2.connect(self.db_url)
    
    def save_channel_config(self, guild_id: str, channel_id: str, 
                           guild_name: str = None, channel_name: str = None) -> bool:
        """Save Discord channel configuration."""
        if self.demo_mode:
            return False
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO discord_channel_config 
                        (guild_id, guild_name, channel_id, channel_name, is_active)
                        VALUES (%s, %s, %s, %s, TRUE)
                        ON CONFLICT (guild_id, channel_id) 
                        DO UPDATE SET 
                            guild_name = EXCLUDED.guild_name,
                            channel_name = EXCLUDED.channel_name,
                            is_active = TRUE,
                            updated_at = NOW()
                    """, (guild_id, guild_name, channel_id, channel_name))
                    conn.commit()
                    return True
        except Exception as e:
            print(f"Error saving channel config: {e}")
            return False
    
    def get_channel_config(self) -> Optional[Dict]:
        """Get active channel configuration."""
        if self.demo_mode:
            return None
        
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT * FROM discord_channel_config 
                        WHERE is_active = TRUE 
                        ORDER BY updated_at DESC LIMIT 1
                    """)
                    result = cur.fetchone()
                    return dict(result) if result else None
        except Exception as e:
            print(f"Error getting channel config: {e}")
            return None
    
    def save_trades(self, trades: List[ParsedTrade], channel_id: str, guild_id: str = None) -> int:
        """Save parsed trades to database."""
        if self.demo_mode or not trades or not channel_id:
            return 0
        
        saved = 0
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    for trade in trades:
                        try:
                            cur.execute("""
                                INSERT INTO discord_trades (
                                    message_id, channel_id, guild_id, author, symbol,
                                    asset_type, action, entry_price, exit_price, quantity,
                                    strike_price, expiration, option_type, profit_loss,
                                    profit_loss_pct, confidence, reasoning, outcome,
                                    raw_message, message_timestamp
                                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                                ON CONFLICT (message_id) DO NOTHING
                            """, (
                                trade.message_id,
                                channel_id,
                                guild_id,
                                trade.author,
                                trade.symbol,
                                trade.asset_type,
                                trade.action,
                                trade.entry_price,
                                trade.exit_price,
                                trade.quantity,
                                trade.strike_price,
                                trade.expiration,
                                trade.option_type,
                                trade.profit_loss,
                                trade.profit_loss_pct,
                                trade.confidence,
                                trade.reasoning,
                                trade.outcome,
                                trade.raw_message[:1000] if trade.raw_message else None,
                                trade.timestamp
                            ))
                            saved += cur.rowcount
                        except Exception as e:
                            print(f"Error saving trade {trade.message_id}: {e}")
                            continue
                    conn.commit()
        except Exception as e:
            print(f"Error saving trades: {e}")
        
        return saved
    
    def get_trades_for_symbol(self, symbol: str, limit: int = 100) -> List[Dict]:
        """Get trades for a specific symbol."""
        if self.demo_mode:
            return []
        
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT * FROM discord_trades 
                        WHERE symbol = %s 
                        ORDER BY message_timestamp DESC 
                        LIMIT %s
                    """, (symbol.upper(), limit))
                    return [dict(row) for row in cur.fetchall()]
        except Exception as e:
            print(f"Error getting trades for {symbol}: {e}")
            return []
    
    def get_all_trades(self, days: int = 365, limit: int = 10000) -> List[Dict]:
        """Get all trades from the past N days."""
        if self.demo_mode:
            return []
        
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT * FROM discord_trades 
                        WHERE message_timestamp > NOW() - INTERVAL '%s days'
                        ORDER BY message_timestamp DESC 
                        LIMIT %s
                    """, (days, limit))
                    return [dict(row) for row in cur.fetchall()]
        except Exception as e:
            print(f"Error getting all trades: {e}")
            return []
    
    def save_pattern(self, pattern: TradingPattern, channel_id: str = None) -> bool:
        """Save a detected trading pattern."""
        if self.demo_mode:
            return False
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO discord_patterns (
                            pattern_type, description, confidence, frequency,
                            symbols, win_rate, avg_profit_pct, time_of_day,
                            day_of_week, channel_id
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        pattern.pattern_type,
                        pattern.description,
                        pattern.confidence,
                        pattern.frequency,
                        pattern.symbols,
                        pattern.win_rate,
                        pattern.avg_profit_pct,
                        pattern.time_of_day,
                        pattern.day_of_week,
                        channel_id
                    ))
                    conn.commit()
                    return True
        except Exception as e:
            print(f"Error saving pattern: {e}")
            return False
    
    def get_active_patterns(self) -> List[Dict]:
        """Get all active trading patterns."""
        if self.demo_mode:
            return []
        
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT * FROM discord_patterns 
                        WHERE is_active = TRUE 
                        ORDER BY confidence DESC
                    """)
                    return [dict(row) for row in cur.fetchall()]
        except Exception as e:
            print(f"Error getting patterns: {e}")
            return []
    
    def save_trader_profile(self, profile: TraderProfile, channel_id: str, 
                           guild_id: str = None) -> Optional[int]:
        """Save a trader profile."""
        if self.demo_mode:
            return None
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO discord_trader_profiles (
                            channel_id, guild_id, total_trades, win_rate,
                            avg_holding_period, preferred_asset_types, preferred_symbols,
                            preferred_actions, risk_tolerance, trading_style,
                            time_preferences, methodology, strengths, weaknesses
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING id
                    """, (
                        channel_id,
                        guild_id,
                        profile.total_trades,
                        profile.win_rate,
                        profile.avg_holding_period,
                        json.dumps(profile.preferred_asset_types),
                        profile.preferred_symbols,
                        json.dumps(profile.preferred_actions),
                        profile.risk_tolerance,
                        profile.trading_style,
                        json.dumps(profile.time_preferences),
                        json.dumps(profile.methodology),
                        profile.strengths,
                        profile.weaknesses
                    ))
                    result = cur.fetchone()
                    conn.commit()
                    return result[0] if result else None
        except Exception as e:
            print(f"Error saving trader profile: {e}")
            return None
    
    def get_latest_profile(self, channel_id: str = None) -> Optional[Dict]:
        """Get the latest trader profile."""
        if self.demo_mode:
            return None
        
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    if channel_id:
                        cur.execute("""
                            SELECT * FROM discord_trader_profiles 
                            WHERE channel_id = %s 
                            ORDER BY analyzed_at DESC LIMIT 1
                        """, (channel_id,))
                    else:
                        cur.execute("""
                            SELECT * FROM discord_trader_profiles 
                            ORDER BY analyzed_at DESC LIMIT 1
                        """)
                    result = cur.fetchone()
                    return dict(result) if result else None
        except Exception as e:
            print(f"Error getting latest profile: {e}")
            return None
    
    def save_symbol_bias(self, symbol: str, channel_id: str, bias_data: Dict) -> bool:
        """Save or update symbol bias."""
        if self.demo_mode:
            return False
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO discord_symbol_bias (
                            symbol, channel_id, bias, confidence, trade_count,
                            win_rate, long_trades, short_trades, long_win_rate,
                            short_win_rate, avg_profit_pct
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (symbol, channel_id) 
                        DO UPDATE SET 
                            bias = EXCLUDED.bias,
                            confidence = EXCLUDED.confidence,
                            trade_count = EXCLUDED.trade_count,
                            win_rate = EXCLUDED.win_rate,
                            long_trades = EXCLUDED.long_trades,
                            short_trades = EXCLUDED.short_trades,
                            long_win_rate = EXCLUDED.long_win_rate,
                            short_win_rate = EXCLUDED.short_win_rate,
                            avg_profit_pct = EXCLUDED.avg_profit_pct,
                            last_updated = NOW()
                    """, (
                        symbol.upper(),
                        channel_id,
                        bias_data.get('bias', 'neutral'),
                        bias_data.get('confidence', 0),
                        bias_data.get('trade_count', 0),
                        bias_data.get('win_rate', 0),
                        bias_data.get('long_trades', 0),
                        bias_data.get('short_trades', 0),
                        bias_data.get('long_win_rate', 0),
                        bias_data.get('short_win_rate', 0),
                        bias_data.get('avg_profit_pct', 0)
                    ))
                    conn.commit()
                    return True
        except Exception as e:
            print(f"Error saving symbol bias: {e}")
            return False
    
    def get_symbol_bias(self, symbol: str) -> Optional[Dict]:
        """Get bias data for a symbol."""
        if self.demo_mode:
            return None
        
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT * FROM discord_symbol_bias 
                        WHERE symbol = %s 
                        ORDER BY last_updated DESC LIMIT 1
                    """, (symbol.upper(),))
                    result = cur.fetchone()
                    return dict(result) if result else None
        except Exception as e:
            print(f"Error getting symbol bias for {symbol}: {e}")
            return None
    
    def save_replication_strategy(self, strategy: Dict, profile_id: int) -> bool:
        """Save a replication strategy."""
        if self.demo_mode:
            return False
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO discord_replication_strategies (
                            profile_id, name, description, directional_bias,
                            preferred_symbols, asset_allocation, risk_level,
                            win_rate_target, entry_rules, exit_rules,
                            risk_management, time_filters, pattern_weights
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        profile_id,
                        strategy.get('name', 'discord_replication'),
                        strategy.get('description'),
                        strategy.get('parameters', {}).get('directional_bias'),
                        strategy.get('parameters', {}).get('preferred_symbols', []),
                        json.dumps(strategy.get('parameters', {}).get('asset_allocation', {})),
                        strategy.get('parameters', {}).get('risk_level'),
                        strategy.get('parameters', {}).get('win_rate_target'),
                        strategy.get('entry_rules', []),
                        strategy.get('exit_rules', []),
                        json.dumps(strategy.get('risk_management', {})),
                        json.dumps(strategy.get('time_filters', {})),
                        json.dumps(strategy.get('pattern_weights', {}))
                    ))
                    conn.commit()
                    return True
        except Exception as e:
            print(f"Error saving replication strategy: {e}")
            return False
    
    def update_sync_stats(self, channel_id: str, messages_synced: int, trades_parsed: int) -> bool:
        """Update sync statistics for a channel."""
        if self.demo_mode:
            return False
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE discord_channel_config 
                        SET messages_synced = messages_synced + %s,
                            trades_parsed = trades_parsed + %s,
                            last_synced_at = NOW(),
                            updated_at = NOW()
                        WHERE channel_id = %s
                    """, (messages_synced, trades_parsed, channel_id))
                    conn.commit()
                    return True
        except Exception as e:
            print(f"Error updating sync stats: {e}")
            return False
