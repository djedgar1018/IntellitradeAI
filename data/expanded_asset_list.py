"""
Expanded Asset Universe - High-Performing S&P 500 Stocks
Based on historical performance data for comprehensive coverage
"""

# Top performing stocks from the provided performance data
EXPANDED_STOCK_UNIVERSE = {
    # Top 20 performers over multiple time periods
    'tier_1_mega_caps': {
        'NVDA': {'name': 'NVIDIA Corp.', 'sector': 'Technology', 'weight': 'High', '5yr_return': '1229%', 'priority': 1},
        'AAPL': {'name': 'Apple Inc.', 'sector': 'Technology', 'weight': 'High', '15yr_return': '19988%', 'priority': 1},
        'MSFT': {'name': 'Microsoft Corp.', 'sector': 'Technology', 'weight': 'High', '15yr_return': '4420%', 'priority': 1},
        'TSLA': {'name': 'Tesla, Inc.', 'sector': 'Consumer Discretionary', 'weight': 'High', '15yr_return': '25322%', 'priority': 1},
        'AMZN': {'name': 'Amazon.com, Inc.', 'sector': 'Consumer Discretionary', 'weight': 'High', '15yr_return': '3517%', 'priority': 1},
    },
    
    'tier_1_growth': {
        'AMD': {'name': 'Advanced Micro Devices, Inc.', 'sector': 'Technology', 'weight': 'Medium', '5yr_return': '1002%', 'priority': 2},
        'AVGO': {'name': 'Broadcom Inc.', 'sector': 'Technology', 'weight': 'Medium', '15yr_return': '20321%', 'priority': 2},
        'NFLX': {'name': 'Netflix, Inc.', 'sector': 'Communication Services', 'weight': 'Medium', '15yr_return': '6607%', 'priority': 2},
        'ANET': {'name': 'Arista Networks, Inc.', 'sector': 'Technology', 'weight': 'Medium', '5yr_return': '884%', 'priority': 2},
        'TPL': {'name': 'Texas Pacific Land Corp.', 'sector': 'Energy', 'weight': 'Medium', '5yr_return': '458%', 'priority': 2},
    },
    
    'tier_2_high_performers': {
        'KLAC': {'name': 'KLA Corp.', 'sector': 'Technology', 'weight': 'Medium', '15yr_return': '5088%', 'priority': 3},
        'MPWR': {'name': 'Monolithic Power Systems, Inc.', 'sector': 'Technology', 'weight': 'Medium', '15yr_return': '5334%', 'priority': 3},
        'CDNS': {'name': 'Cadence Design Systems, Inc.', 'sector': 'Technology', 'weight': 'Medium', '15yr_return': '5016%', 'priority': 3},
        'SNPS': {'name': 'Synopsys, Inc.', 'sector': 'Technology', 'weight': 'Medium', '5yr_return': '424%', 'priority': 3},
        'REGN': {'name': 'Regeneron Pharmaceuticals, Inc.', 'sector': 'Healthcare', 'weight': 'Medium', '15yr_return': '5103%', 'priority': 3},
    },
    
    'tier_2_diversified': {
        'FICO': {'name': 'Fair Isaac Corp.', 'sector': 'Technology', 'weight': 'Medium', '15yr_return': '6578%', 'priority': 3},
        'LRCX': {'name': 'Lam Research Corp.', 'sector': 'Technology', 'weight': 'Medium', '5yr_return': '458%', 'priority': 3},
        'PWR': {'name': 'Quanta Services, Inc.', 'sector': 'Industrials', 'weight': 'Medium', '5yr_return': '643%', 'priority': 3},
        'DECK': {'name': 'Deckers Outdoor Corp.', 'sector': 'Consumer Discretionary', 'weight': 'Medium', '15yr_return': '6578%', 'priority': 3},
        'AXON': {'name': 'Axon Enterprise, Inc.', 'sector': 'Industrials', 'weight': 'Medium', '15yr_return': '6380%', 'priority': 3},
    },
    
    'tier_3_specialized': {
        'SMCI': {'name': 'Super Micro Computer, Inc.', 'sector': 'Technology', 'weight': 'Low', '5yr_return': '1435%', 'priority': 4},
        'VISTRA': {'name': 'Vistra Corp.', 'sector': 'Utilities', 'weight': 'Low', '5yr_return': '1002%', 'priority': 4},
        'TRGP': {'name': 'Targa Resources Corp.', 'sector': 'Energy', 'weight': 'Low', '5yr_return': '943%', 'priority': 4},
        'HWM': {'name': 'Howmet Aerospace, Inc.', 'sector': 'Industrials', 'weight': 'Low', '5yr_return': '879%', 'priority': 4},
        'JBIL': {'name': 'Jabil, Inc.', 'sector': 'Technology', 'weight': 'Low', '5yr_return': '455%', 'priority': 4},
    },
    
    'tier_3_emerging': {
        'URI': {'name': 'United Rentals, Inc.', 'sector': 'Industrials', 'weight': 'Low', '15yr_return': '8224%', 'priority': 4},
        'NOW': {'name': 'ServiceNow, Inc.', 'sector': 'Technology', 'weight': 'Low', '5yr_return': '355%', 'priority': 4},
        'TYL': {'name': 'Tyler Technologies, Inc.', 'sector': 'Technology', 'weight': 'Low', '15yr_return': '5103%', 'priority': 4},
        'MNST': {'name': 'Monster Beverage Corp.', 'sector': 'Consumer Staples', 'weight': 'Low', '15yr_return': '5016%', 'priority': 4},
        'CRM': {'name': 'Salesforce, Inc.', 'sector': 'Technology', 'weight': 'Low', '15yr_return': '3902%', 'priority': 4},
    }
}

# Cryptocurrency universe for comprehensive coverage
CRYPTO_UNIVERSE = {
    'tier_1_major': {
        'BTC': {'name': 'Bitcoin', 'market_cap': 'Large', 'volatility': 'High', 'priority': 1},
        'ETH': {'name': 'Ethereum', 'market_cap': 'Large', 'volatility': 'High', 'priority': 1},
        'BNB': {'name': 'Binance Coin', 'market_cap': 'Large', 'volatility': 'High', 'priority': 1},
    },
    'tier_2_alternative': {
        'ADA': {'name': 'Cardano', 'market_cap': 'Medium', 'volatility': 'Very High', 'priority': 2},
        'SOL': {'name': 'Solana', 'market_cap': 'Medium', 'volatility': 'Very High', 'priority': 2},
        'DOT': {'name': 'Polkadot', 'market_cap': 'Medium', 'volatility': 'Very High', 'priority': 2},
    },
    'tier_3_emerging': {
        'AVAX': {'name': 'Avalanche', 'market_cap': 'Medium', 'volatility': 'Very High', 'priority': 3},
        'MATIC': {'name': 'Polygon', 'market_cap': 'Medium', 'volatility': 'Very High', 'priority': 3},
        'ATOM': {'name': 'Cosmos', 'market_cap': 'Small', 'volatility': 'Very High', 'priority': 3},
    }
}

def get_priority_assets(max_stocks=10, max_crypto=5):
    """Get highest priority assets for analysis"""
    stocks = []
    crypto = []
    
    # Collect stocks by priority
    for tier_name, tier_stocks in EXPANDED_STOCK_UNIVERSE.items():
        for symbol, data in tier_stocks.items():
            stocks.append((symbol, data['priority'], data))
    
    # Collect crypto by priority  
    for tier_name, tier_crypto in CRYPTO_UNIVERSE.items():
        for symbol, data in tier_crypto.items():
            crypto.append((symbol, data['priority'], data))
    
    # Sort by priority and return top assets
    stocks.sort(key=lambda x: x[1])
    crypto.sort(key=lambda x: x[1])
    
    return (
        [stock[0] for stock in stocks[:max_stocks]],
        [c[0] for c in crypto[:max_crypto]]
    )

def get_sector_diversification():
    """Get assets grouped by sector for diversified analysis"""
    sectors = {}
    
    for tier_name, tier_stocks in EXPANDED_STOCK_UNIVERSE.items():
        for symbol, data in tier_stocks.items():
            sector = data['sector']
            if sector not in sectors:
                sectors[sector] = []
            sectors[sector].append(symbol)
    
    return sectors

def get_performance_leaders():
    """Get top performers by different time periods"""
    leaders = {
        '5_year_leaders': [],
        '15_year_leaders': [],
        'recent_momentum': []
    }
    
    # Extract 5-year leaders
    for tier_name, tier_stocks in EXPANDED_STOCK_UNIVERSE.items():
        for symbol, data in tier_stocks.items():
            if '5yr_return' in data:
                return_pct = float(data['5yr_return'].replace('%', ''))
                if return_pct > 400:  # 400%+ returns
                    leaders['5_year_leaders'].append((symbol, return_pct))
    
    # Extract 15-year leaders
    for tier_name, tier_stocks in EXPANDED_STOCK_UNIVERSE.items():
        for symbol, data in tier_stocks.items():
            if '15yr_return' in data:
                return_pct = float(data['15yr_return'].replace('%', ''))
                if return_pct > 5000:  # 5000%+ returns
                    leaders['15_year_leaders'].append((symbol, return_pct))
    
    # Sort by performance
    leaders['5_year_leaders'].sort(key=lambda x: x[1], reverse=True)
    leaders['15_year_leaders'].sort(key=lambda x: x[1], reverse=True)
    
    return leaders

# Asset allocation suggestions based on risk profiles
RISK_BASED_ALLOCATIONS = {
    'conservative': {
        'description': 'Focus on established large-caps with proven track records',
        'stock_allocation': 0.70,
        'crypto_allocation': 0.10,
        'cash_allocation': 0.20,
        'preferred_stocks': ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'FICO'],
        'preferred_crypto': ['BTC', 'ETH']
    },
    
    'moderate': {
        'description': 'Balanced approach with growth and stability',
        'stock_allocation': 0.60,
        'crypto_allocation': 0.20,
        'cash_allocation': 0.20,
        'preferred_stocks': ['NVDA', 'AMD', 'TSLA', 'AVGO', 'NFLX', 'ANET', 'KLAC'],
        'preferred_crypto': ['BTC', 'ETH', 'BNB']
    },
    
    'aggressive': {
        'description': 'High-growth focus with higher volatility tolerance',
        'stock_allocation': 0.50,
        'crypto_allocation': 0.40,
        'cash_allocation': 0.10,
        'preferred_stocks': ['SMCI', 'VISTRA', 'TRGP', 'AMD', 'TSLA', 'ANET', 'CDNS'],
        'preferred_crypto': ['BTC', 'ETH', 'ADA', 'SOL', 'AVAX']
    }
}

def get_risk_based_portfolio(risk_level='moderate'):
    """Get suggested portfolio allocation based on risk tolerance"""
    if risk_level not in RISK_BASED_ALLOCATIONS:
        risk_level = 'moderate'
    
    allocation = RISK_BASED_ALLOCATIONS[risk_level]
    
    return {
        'risk_profile': risk_level,
        'description': allocation['description'],
        'allocations': {
            'stocks': allocation['stock_allocation'],
            'crypto': allocation['crypto_allocation'], 
            'cash': allocation['cash_allocation']
        },
        'recommended_stocks': allocation['preferred_stocks'],
        'recommended_crypto': allocation['preferred_crypto'],
        'total_assets': len(allocation['preferred_stocks']) + len(allocation['preferred_crypto'])
    }