-- Secure Trading Bot Database Schema
-- Built for blockchain integration with user accounts, portfolios, and trading logs

-- Users table with secure authentication
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    salt VARCHAR(255) NOT NULL,
    is_2fa_enabled BOOLEAN DEFAULT FALSE,
    two_fa_secret VARCHAR(255),
    email_verified BOOLEAN DEFAULT FALSE,
    email_verification_token VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP,
    login_attempts INTEGER DEFAULT 0,
    locked_until TIMESTAMP,
    api_key_hash VARCHAR(255),
    api_secret_hash VARCHAR(255),
    is_active BOOLEAN DEFAULT TRUE
);

-- Crypto wallets for each user
CREATE TABLE crypto_wallets (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    wallet_address VARCHAR(255) NOT NULL,
    private_key_encrypted TEXT NOT NULL, -- AES-256 encrypted
    wallet_type VARCHAR(20) NOT NULL, -- 'ethereum', 'bitcoin', etc.
    balance DECIMAL(20,8) DEFAULT 0.00,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    mnemonic_encrypted TEXT, -- Encrypted seed phrase
    derivation_path VARCHAR(100) DEFAULT "m/44'/60'/0'/0/0"
);

-- User portfolios (separate for stocks and crypto)
CREATE TABLE portfolios (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    portfolio_type VARCHAR(10) NOT NULL, -- 'stock' or 'crypto'
    name VARCHAR(100) NOT NULL,
    starting_capital DECIMAL(15,2) DEFAULT 0.00,
    current_value DECIMAL(15,2) DEFAULT 0.00,
    total_invested DECIMAL(15,2) DEFAULT 0.00,
    realized_pnl DECIMAL(15,2) DEFAULT 0.00,
    unrealized_pnl DECIMAL(15,2) DEFAULT 0.00,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Holdings within portfolios
CREATE TABLE holdings (
    id SERIAL PRIMARY KEY,
    portfolio_id INTEGER REFERENCES portfolios(id) ON DELETE CASCADE,
    symbol VARCHAR(20) NOT NULL,
    quantity DECIMAL(20,8) NOT NULL,
    average_price DECIMAL(15,8) NOT NULL,
    current_price DECIMAL(15,8) DEFAULT 0.00,
    market_value DECIMAL(15,2) DEFAULT 0.00,
    unrealized_pnl DECIMAL(15,2) DEFAULT 0.00,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Trading signals and AI predictions
CREATE TABLE trading_signals (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    symbol VARCHAR(20) NOT NULL,
    signal_type VARCHAR(10) NOT NULL, -- 'BUY', 'SELL', 'HOLD', 'DCA_IN', 'DCA_OUT'
    confidence_level DECIMAL(3,2) NOT NULL,
    risk_level VARCHAR(10) NOT NULL, -- 'Low', 'Medium', 'High'
    entry_price DECIMAL(15,8),
    target_price DECIMAL(15,8),
    stop_loss_price DECIMAL(15,8),
    current_price DECIMAL(15,8) NOT NULL,
    model_used VARCHAR(50), -- 'RandomForest', 'XGBoost', 'PatternRecognition'
    reasoning TEXT,
    chart_pattern VARCHAR(100), -- Detected chart pattern
    pattern_confidence DECIMAL(3,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

-- Trade executions and tracking
CREATE TABLE trades (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    portfolio_id INTEGER REFERENCES portfolios(id) ON DELETE CASCADE,
    signal_id INTEGER REFERENCES trading_signals(id),
    symbol VARCHAR(20) NOT NULL,
    trade_type VARCHAR(10) NOT NULL, -- 'BUY', 'SELL'
    quantity DECIMAL(20,8) NOT NULL,
    entry_price DECIMAL(15,8) NOT NULL,
    exit_price DECIMAL(15,8),
    executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    closed_at TIMESTAMP,
    status VARCHAR(20) DEFAULT 'OPEN', -- 'OPEN', 'CLOSED', 'CANCELLED'
    pnl DECIMAL(15,2) DEFAULT 0.00,
    fees DECIMAL(15,2) DEFAULT 0.00,
    trade_source VARCHAR(20) DEFAULT 'AI_BOT', -- 'AI_BOT', 'MANUAL'
    blockchain_tx_hash VARCHAR(255), -- For crypto trades
    gas_fee DECIMAL(15,8) -- For blockchain transactions
);

-- Chart pattern recognition results
CREATE TABLE chart_patterns (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    pattern_type VARCHAR(100) NOT NULL,
    confidence DECIMAL(3,2) NOT NULL,
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    timeframe VARCHAR(10) NOT NULL, -- '1m', '5m', '1h', '1d'
    image_data BYTEA, -- Store chart image for analysis
    bounding_box JSON, -- Pattern location coordinates
    predicted_direction VARCHAR(10), -- 'UP', 'DOWN', 'SIDEWAYS'
    success_rate DECIMAL(3,2), -- Historical success rate of this pattern
    signal_generated BOOLEAN DEFAULT FALSE,
    outcome VARCHAR(20), -- 'SUCCESS', 'FAIL', 'PENDING' (for tracking accuracy)
    outcome_measured_at TIMESTAMP
);

-- AI model performance tracking
CREATE TABLE model_performance (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(50) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    prediction_date TIMESTAMP NOT NULL,
    predicted_direction VARCHAR(10) NOT NULL,
    predicted_price DECIMAL(15,8),
    actual_price DECIMAL(15,8),
    actual_direction VARCHAR(10),
    accuracy_score DECIMAL(3,2),
    timeframe_hours INTEGER DEFAULT 24, -- How far ahead was the prediction
    confidence DECIMAL(3,2),
    success BOOLEAN,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Security audit log
CREATE TABLE security_logs (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    event_type VARCHAR(50) NOT NULL, -- 'LOGIN', 'LOGOUT', 'API_ACCESS', '2FA_VERIFY'
    ip_address INET,
    user_agent TEXT,
    success BOOLEAN NOT NULL,
    details JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- User preferences and settings
CREATE TABLE user_settings (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    risk_tolerance VARCHAR(20) DEFAULT 'MEDIUM', -- 'LOW', 'MEDIUM', 'HIGH'
    max_position_size DECIMAL(5,2) DEFAULT 10.00, -- Percentage of portfolio
    auto_trade_enabled BOOLEAN DEFAULT FALSE,
    notification_preferences JSON,
    preferred_exchanges JSON, -- List of preferred exchanges for crypto
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Session management for security
CREATE TABLE user_sessions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    session_token VARCHAR(255) NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

-- Indexes for performance
CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_portfolios_user_id ON portfolios(user_id);
CREATE INDEX idx_trades_user_id ON trades(user_id);
CREATE INDEX idx_trades_symbol ON trades(symbol);
CREATE INDEX idx_trading_signals_user_id ON trading_signals(user_id);
CREATE INDEX idx_trading_signals_symbol ON trading_signals(symbol);
CREATE INDEX idx_trading_signals_created_at ON trading_signals(created_at);
CREATE INDEX idx_chart_patterns_symbol ON chart_patterns(symbol);
CREATE INDEX idx_chart_patterns_detected_at ON chart_patterns(detected_at);
CREATE INDEX idx_model_performance_model_symbol ON model_performance(model_name, symbol);
CREATE INDEX idx_security_logs_user_id ON security_logs(user_id);
CREATE INDEX idx_user_sessions_token ON user_sessions(session_token);

-- Constraints
ALTER TABLE crypto_wallets ADD CONSTRAINT unique_user_wallet_type UNIQUE(user_id, wallet_type);
ALTER TABLE portfolios ADD CONSTRAINT unique_user_portfolio_name UNIQUE(user_id, name);