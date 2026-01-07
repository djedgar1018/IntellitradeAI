-- Discord Integration Tables for IntelliTradeAI
-- Stores Discord trade history, learned patterns, and trading methodology

-- Discord trade messages parsed from conversations
CREATE TABLE IF NOT EXISTS discord_trades (
    id SERIAL PRIMARY KEY,
    message_id VARCHAR(255) UNIQUE NOT NULL,
    channel_id VARCHAR(255) NOT NULL,
    guild_id VARCHAR(255),
    author VARCHAR(255) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    asset_type VARCHAR(20) NOT NULL,  -- 'crypto', 'stock', 'option'
    action VARCHAR(20) NOT NULL,  -- 'buy', 'sell', 'call', 'put'
    entry_price DECIMAL(15,8),
    exit_price DECIMAL(15,8),
    quantity DECIMAL(20,8),
    strike_price DECIMAL(15,8),
    expiration VARCHAR(50),
    option_type VARCHAR(10),  -- 'call', 'put'
    profit_loss DECIMAL(15,2),
    profit_loss_pct DECIMAL(8,2),
    confidence DECIMAL(5,2),
    reasoning TEXT,
    outcome VARCHAR(20),  -- 'win', 'loss', null
    raw_message TEXT,
    message_timestamp TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Learned trading patterns from Discord analysis
CREATE TABLE IF NOT EXISTS discord_patterns (
    id SERIAL PRIMARY KEY,
    pattern_type VARCHAR(50) NOT NULL,  -- 'directional_bias', 'sector_focus', 'momentum_following', 'contrarian'
    description TEXT NOT NULL,
    confidence DECIMAL(5,2) NOT NULL,
    frequency INTEGER DEFAULT 0,
    symbols TEXT[],  -- Array of symbols related to this pattern
    win_rate DECIMAL(5,2),
    avg_profit_pct DECIMAL(8,2),
    time_of_day VARCHAR(50),
    day_of_week VARCHAR(20),
    channel_id VARCHAR(255),
    analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

-- Trader profile learned from Discord
CREATE TABLE IF NOT EXISTS discord_trader_profiles (
    id SERIAL PRIMARY KEY,
    channel_id VARCHAR(255) NOT NULL,
    guild_id VARCHAR(255),
    total_trades INTEGER DEFAULT 0,
    win_rate DECIMAL(5,2),
    avg_holding_period DECIMAL(8,2),
    preferred_asset_types JSONB,  -- {'crypto': 60, 'stock': 30, 'option': 10}
    preferred_symbols TEXT[],
    preferred_actions JSONB,  -- {'buy': 70, 'sell': 20, 'call': 10}
    risk_tolerance VARCHAR(20),  -- 'conservative', 'moderate', 'aggressive'
    trading_style VARCHAR(50),  -- 'bullish_long', 'aggressive_calls', etc.
    time_preferences JSONB,  -- peak hours and days
    methodology JSONB,  -- entry/exit criteria, risk management
    strengths TEXT[],
    weaknesses TEXT[],
    analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Replication strategy derived from Discord analysis
CREATE TABLE IF NOT EXISTS discord_replication_strategies (
    id SERIAL PRIMARY KEY,
    profile_id INTEGER REFERENCES discord_trader_profiles(id) ON DELETE CASCADE,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    directional_bias VARCHAR(20),  -- 'long', 'short', 'neutral'
    preferred_symbols TEXT[],
    asset_allocation JSONB,
    risk_level VARCHAR(20),
    win_rate_target DECIMAL(5,2),
    entry_rules TEXT[],
    exit_rules TEXT[],
    risk_management JSONB,  -- max_position_size, stop_loss, take_profit
    time_filters JSONB,  -- preferred hours and days
    pattern_weights JSONB,  -- weights for different patterns
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Discord channel configuration
CREATE TABLE IF NOT EXISTS discord_channel_config (
    id SERIAL PRIMARY KEY,
    guild_id VARCHAR(255) NOT NULL,
    guild_name VARCHAR(255),
    channel_id VARCHAR(255) NOT NULL,
    channel_name VARCHAR(255),
    is_active BOOLEAN DEFAULT TRUE,
    last_synced_at TIMESTAMP,
    messages_synced INTEGER DEFAULT 0,
    trades_parsed INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(guild_id, channel_id)
);

-- Symbol bias learned from Discord
CREATE TABLE IF NOT EXISTS discord_symbol_bias (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    channel_id VARCHAR(255) NOT NULL,
    bias VARCHAR(20) NOT NULL,  -- 'bullish', 'bearish', 'neutral'
    confidence DECIMAL(5,2),
    trade_count INTEGER DEFAULT 0,
    win_rate DECIMAL(5,2),
    long_trades INTEGER DEFAULT 0,
    short_trades INTEGER DEFAULT 0,
    long_win_rate DECIMAL(5,2),
    short_win_rate DECIMAL(5,2),
    avg_profit_pct DECIMAL(8,2),
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, channel_id)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_discord_trades_symbol ON discord_trades(symbol);
CREATE INDEX IF NOT EXISTS idx_discord_trades_channel ON discord_trades(channel_id);
CREATE INDEX IF NOT EXISTS idx_discord_trades_timestamp ON discord_trades(message_timestamp);
CREATE INDEX IF NOT EXISTS idx_discord_trades_outcome ON discord_trades(outcome);
CREATE INDEX IF NOT EXISTS idx_discord_patterns_type ON discord_patterns(pattern_type);
CREATE INDEX IF NOT EXISTS idx_discord_patterns_channel ON discord_patterns(channel_id);
CREATE INDEX IF NOT EXISTS idx_discord_profiles_channel ON discord_trader_profiles(channel_id);
CREATE INDEX IF NOT EXISTS idx_discord_symbol_bias_symbol ON discord_symbol_bias(symbol);
CREATE INDEX IF NOT EXISTS idx_discord_symbol_bias_channel ON discord_symbol_bias(channel_id);
