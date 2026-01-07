-- Paper Trading System Schema
-- For options paper trading with risk management and improvement loops

-- Paper Trading Sessions
CREATE TABLE IF NOT EXISTS paper_trading_sessions (
    session_id VARCHAR(36) PRIMARY KEY,
    starting_balance DECIMAL(15, 2) NOT NULL DEFAULT 100000.00,
    current_balance DECIMAL(15, 2) NOT NULL DEFAULT 100000.00,
    target_balance DECIMAL(15, 2) NOT NULL DEFAULT 200000.00,
    peak_balance DECIMAL(15, 2) NOT NULL DEFAULT 100000.00,
    max_drawdown_limit DECIMAL(5, 2) NOT NULL DEFAULT 30.00,
    current_drawdown DECIMAL(5, 2) NOT NULL DEFAULT 0.00,
    status VARCHAR(20) NOT NULL DEFAULT 'active',
    strategy_version INTEGER NOT NULL DEFAULT 1,
    total_trades INTEGER NOT NULL DEFAULT 0,
    winning_trades INTEGER NOT NULL DEFAULT 0,
    losing_trades INTEGER NOT NULL DEFAULT 0,
    total_realized_pnl DECIMAL(15, 2) NOT NULL DEFAULT 0.00,
    total_unrealized_pnl DECIMAL(15, 2) NOT NULL DEFAULT 0.00,
    started_at TIMESTAMP DEFAULT NOW(),
    ended_at TIMESTAMP,
    end_reason VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Options Positions for Paper Trading
CREATE TABLE IF NOT EXISTS paper_options_positions (
    position_id VARCHAR(36) PRIMARY KEY,
    session_id VARCHAR(36) NOT NULL REFERENCES paper_trading_sessions(session_id),
    symbol VARCHAR(20) NOT NULL,
    option_type VARCHAR(10) NOT NULL,
    strike_price DECIMAL(15, 2) NOT NULL,
    expiration_date DATE NOT NULL,
    contracts INTEGER NOT NULL,
    entry_price DECIMAL(15, 4) NOT NULL,
    current_price DECIMAL(15, 4),
    entry_delta DECIMAL(10, 4),
    entry_gamma DECIMAL(10, 4),
    entry_theta DECIMAL(10, 4),
    entry_vega DECIMAL(10, 4),
    current_delta DECIMAL(10, 4),
    current_gamma DECIMAL(10, 4),
    current_theta DECIMAL(10, 4),
    current_vega DECIMAL(10, 4),
    implied_volatility DECIMAL(10, 4),
    unrealized_pnl DECIMAL(15, 2) DEFAULT 0.00,
    realized_pnl DECIMAL(15, 2) DEFAULT 0.00,
    ai_signal VARCHAR(10),
    ai_confidence DECIMAL(5, 2),
    status VARCHAR(20) NOT NULL DEFAULT 'open',
    opened_at TIMESTAMP DEFAULT NOW(),
    closed_at TIMESTAMP,
    close_reason VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Risk Snapshots for tracking portfolio over time
CREATE TABLE IF NOT EXISTS paper_trading_snapshots (
    snapshot_id VARCHAR(36) PRIMARY KEY,
    session_id VARCHAR(36) NOT NULL REFERENCES paper_trading_sessions(session_id),
    timestamp TIMESTAMP DEFAULT NOW(),
    portfolio_value DECIMAL(15, 2) NOT NULL,
    cash_balance DECIMAL(15, 2) NOT NULL,
    positions_value DECIMAL(15, 2) NOT NULL,
    unrealized_pnl DECIMAL(15, 2) NOT NULL,
    realized_pnl DECIMAL(15, 2) NOT NULL,
    drawdown_percent DECIMAL(5, 2) NOT NULL,
    total_delta DECIMAL(15, 4),
    total_gamma DECIMAL(15, 4),
    total_theta DECIMAL(15, 4),
    total_vega DECIMAL(15, 4),
    open_positions INTEGER NOT NULL,
    notes TEXT
);

-- Improvement Logs for strategy adjustments
CREATE TABLE IF NOT EXISTS paper_trading_improvements (
    improvement_id VARCHAR(36) PRIMARY KEY,
    session_id VARCHAR(36) NOT NULL REFERENCES paper_trading_sessions(session_id),
    trigger_reason VARCHAR(50) NOT NULL,
    pre_improvement_metrics JSONB,
    analysis_summary TEXT,
    improvements_made JSONB,
    strategy_changes TEXT,
    new_strategy_version INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Trades within paper trading sessions
CREATE TABLE IF NOT EXISTS paper_trades (
    trade_id VARCHAR(36) PRIMARY KEY,
    session_id VARCHAR(36) NOT NULL REFERENCES paper_trading_sessions(session_id),
    position_id VARCHAR(36) REFERENCES paper_options_positions(position_id),
    symbol VARCHAR(20) NOT NULL,
    action VARCHAR(10) NOT NULL,
    option_type VARCHAR(10) NOT NULL,
    strike_price DECIMAL(15, 2) NOT NULL,
    expiration_date DATE NOT NULL,
    contracts INTEGER NOT NULL,
    price DECIMAL(15, 4) NOT NULL,
    total_value DECIMAL(15, 2) NOT NULL,
    fees DECIMAL(10, 2) DEFAULT 0.00,
    ai_signal VARCHAR(10),
    ai_confidence DECIMAL(5, 2),
    signal_reasoning TEXT,
    executed_at TIMESTAMP DEFAULT NOW()
);

-- Strategy Configuration
CREATE TABLE IF NOT EXISTS paper_trading_strategy (
    strategy_id VARCHAR(36) PRIMARY KEY,
    version INTEGER NOT NULL,
    name VARCHAR(100),
    position_size_percent DECIMAL(5, 2) DEFAULT 5.00,
    max_positions INTEGER DEFAULT 10,
    min_confidence DECIMAL(5, 2) DEFAULT 70.00,
    stop_loss_percent DECIMAL(5, 2) DEFAULT 25.00,
    take_profit_percent DECIMAL(5, 2) DEFAULT 50.00,
    max_days_to_expiry INTEGER DEFAULT 45,
    min_days_to_expiry INTEGER DEFAULT 7,
    delta_range_min DECIMAL(5, 2) DEFAULT 0.30,
    delta_range_max DECIMAL(5, 2) DEFAULT 0.70,
    allowed_symbols TEXT[],
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_paper_positions_session ON paper_options_positions(session_id);
CREATE INDEX IF NOT EXISTS idx_paper_positions_status ON paper_options_positions(status);
CREATE INDEX IF NOT EXISTS idx_paper_snapshots_session ON paper_trading_snapshots(session_id);
CREATE INDEX IF NOT EXISTS idx_paper_trades_session ON paper_trades(session_id);
CREATE INDEX IF NOT EXISTS idx_paper_improvements_session ON paper_trading_improvements(session_id);
