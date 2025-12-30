"""
IntelliTradeAI Comprehensive Asset Configuration
Top 100 Cryptocurrencies with Sector Categorization
Complete Stock Market Sectors, Industries, and Indices
"""
from typing import Dict, List, Optional

class CryptoAssets:
    """Top 100 cryptocurrencies organized by sector"""
    
    SECTORS = {
        "Layer 1 (L1)": {
            "description": "Base blockchain networks that process transactions",
            "risk_level": "Medium-High",
            "assets": [
                {"symbol": "BTC", "name": "Bitcoin", "rank": 1},
                {"symbol": "ETH", "name": "Ethereum", "rank": 2},
                {"symbol": "SOL", "name": "Solana", "rank": 5},
                {"symbol": "BNB", "name": "BNB", "rank": 4},
                {"symbol": "XRP", "name": "XRP", "rank": 6},
                {"symbol": "ADA", "name": "Cardano", "rank": 10},
                {"symbol": "AVAX", "name": "Avalanche", "rank": 12},
                {"symbol": "DOT", "name": "Polkadot", "rank": 15},
                {"symbol": "TRX", "name": "TRON", "rank": 9},
                {"symbol": "NEAR", "name": "NEAR Protocol", "rank": 18},
                {"symbol": "ICP", "name": "Internet Computer", "rank": 25},
                {"symbol": "APT", "name": "Aptos", "rank": 28},
                {"symbol": "SUI", "name": "Sui", "rank": 20},
                {"symbol": "SEI", "name": "Sei", "rank": 45},
                {"symbol": "ATOM", "name": "Cosmos", "rank": 30},
                {"symbol": "ALGO", "name": "Algorand", "rank": 55},
                {"symbol": "FTM", "name": "Fantom", "rank": 60},
                {"symbol": "HBAR", "name": "Hedera", "rank": 22},
                {"symbol": "EOS", "name": "EOS", "rank": 70},
                {"symbol": "XLM", "name": "Stellar", "rank": 35}
            ]
        },
        "Layer 2 (L2)": {
            "description": "Scaling solutions built on top of Layer 1 networks",
            "risk_level": "Medium-High",
            "assets": [
                {"symbol": "MATIC", "name": "Polygon", "rank": 16},
                {"symbol": "ARB", "name": "Arbitrum", "rank": 40},
                {"symbol": "OP", "name": "Optimism", "rank": 42},
                {"symbol": "IMX", "name": "Immutable", "rank": 38},
                {"symbol": "MNT", "name": "Mantle", "rank": 50},
                {"symbol": "STRK", "name": "Starknet", "rank": 65},
                {"symbol": "ZK", "name": "zkSync", "rank": 80},
                {"symbol": "METIS", "name": "Metis", "rank": 90}
            ]
        },
        "Stablecoins": {
            "description": "Cryptocurrencies pegged to fiat currencies",
            "risk_level": "Low",
            "assets": [
                {"symbol": "USDT", "name": "Tether", "rank": 3},
                {"symbol": "USDC", "name": "USD Coin", "rank": 7},
                {"symbol": "DAI", "name": "Dai", "rank": 24},
                {"symbol": "FDUSD", "name": "First Digital USD", "rank": 11},
                {"symbol": "TUSD", "name": "TrueUSD", "rank": 85},
                {"symbol": "USDD", "name": "USDD", "rank": 75}
            ]
        },
        "DeFi": {
            "description": "Decentralized finance protocols and tokens",
            "risk_level": "High",
            "assets": [
                {"symbol": "LINK", "name": "Chainlink", "rank": 13},
                {"symbol": "UNI", "name": "Uniswap", "rank": 23},
                {"symbol": "AAVE", "name": "Aave", "rank": 32},
                {"symbol": "MKR", "name": "Maker", "rank": 34},
                {"symbol": "LDO", "name": "Lido DAO", "rank": 36},
                {"symbol": "INJ", "name": "Injective", "rank": 27},
                {"symbol": "CRV", "name": "Curve DAO", "rank": 75},
                {"symbol": "SNX", "name": "Synthetix", "rank": 80},
                {"symbol": "COMP", "name": "Compound", "rank": 95},
                {"symbol": "SUSHI", "name": "SushiSwap", "rank": 98},
                {"symbol": "1INCH", "name": "1inch", "rank": 92},
                {"symbol": "DYDX", "name": "dYdX", "rank": 85},
                {"symbol": "JUP", "name": "Jupiter", "rank": 48},
                {"symbol": "PENDLE", "name": "Pendle", "rank": 72}
            ]
        },
        "AI Tokens": {
            "description": "Tokens related to artificial intelligence and machine learning infrastructure",
            "risk_level": "Very High",
            "assets": [
                {"symbol": "FET", "name": "Fetch.ai", "rank": 29},
                {"symbol": "RNDR", "name": "Render", "rank": 26},
                {"symbol": "AGIX", "name": "SingularityNET", "rank": 68},
                {"symbol": "OCEAN", "name": "Ocean Protocol", "rank": 88},
                {"symbol": "TAO", "name": "Bittensor", "rank": 21},
                {"symbol": "AKT", "name": "Akash Network", "rank": 58},
                {"symbol": "ARKM", "name": "Arkham", "rank": 78},
                {"symbol": "WLD", "name": "Worldcoin", "rank": 52},
                {"symbol": "GRASS", "name": "Grass", "rank": 100},
                {"symbol": "IO", "name": "io.net", "rank": 115},
                {"symbol": "NOS", "name": "Nosana", "rank": 175},
                {"symbol": "PRIME", "name": "Echelon Prime", "rank": 130}
            ]
        },
        "AI Agent Tokens": {
            "description": "Tokens powering autonomous AI agents and decentralized AI systems",
            "risk_level": "Extreme",
            "assets": [
                {"symbol": "VIRTUAL", "name": "Virtuals Protocol", "rank": 35},
                {"symbol": "AI16Z", "name": "ai16z", "rank": 110},
                {"symbol": "GOAT", "name": "Goatseus Maximus", "rank": 125},
                {"symbol": "AIXBT", "name": "aixbt by Virtuals", "rank": 140},
                {"symbol": "ZEREBRO", "name": "Zerebro", "rank": 155},
                {"symbol": "GRIFFAIN", "name": "Griffain", "rank": 170},
                {"symbol": "LUNA2", "name": "Luna by Virtuals", "rank": 185},
                {"symbol": "ARC", "name": "Arc", "rank": 190},
                {"symbol": "SWARMS", "name": "Swarms", "rank": 195},
                {"symbol": "PAAL", "name": "PAAL AI", "rank": 160},
                {"symbol": "CGPT", "name": "ChainGPT", "rank": 165},
                {"symbol": "AGRS", "name": "Agoras", "rank": 200},
                {"symbol": "AI", "name": "Sleepless AI", "rank": 95},
                {"symbol": "FARTCOIN", "name": "Fartcoin", "rank": 85}
            ]
        },
        "Meme Coins": {
            "description": "Tokens originating from internet memes and culture",
            "risk_level": "Extreme",
            "assets": [
                {"symbol": "DOGE", "name": "Dogecoin", "rank": 8},
                {"symbol": "SHIB", "name": "Shiba Inu", "rank": 14},
                {"symbol": "PEPE", "name": "Pepe", "rank": 19},
                {"symbol": "WIF", "name": "dogwifhat", "rank": 33},
                {"symbol": "BONK", "name": "Bonk", "rank": 46},
                {"symbol": "FLOKI", "name": "FLOKI", "rank": 54},
                {"symbol": "MEME", "name": "Memecoin", "rank": 82},
                {"symbol": "ELON", "name": "Dogelon Mars", "rank": 97},
                {"symbol": "BRETT", "name": "Brett", "rank": 65},
                {"symbol": "POPCAT", "name": "Popcat", "rank": 70},
                {"symbol": "MOG", "name": "Mog Coin", "rank": 75},
                {"symbol": "TURBO", "name": "Turbo", "rank": 135},
                {"symbol": "NEIRO", "name": "First Neiro", "rank": 80},
                {"symbol": "SPX", "name": "SPX6900", "rank": 90},
                {"symbol": "PNUT", "name": "Peanut the Squirrel", "rank": 85},
                {"symbol": "MYRO", "name": "Myro", "rank": 145},
                {"symbol": "MEW", "name": "cat in a dogs world", "rank": 95},
                {"symbol": "BABYDOGE", "name": "Baby Doge Coin", "rank": 100},
                {"symbol": "COQ", "name": "Coq Inu", "rank": 150},
                {"symbol": "BOME", "name": "Book of Meme", "rank": 60},
                {"symbol": "SLERF", "name": "Slerf", "rank": 155},
                {"symbol": "PONKE", "name": "Ponke", "rank": 160},
                {"symbol": "GIGA", "name": "GigaChad", "rank": 165},
                {"symbol": "MOODENG", "name": "Moo Deng", "rank": 120}
            ]
        },
        "NFT Projects": {
            "description": "Tokens associated with major NFT platforms and collections",
            "risk_level": "Very High",
            "assets": [
                {"symbol": "APE", "name": "ApeCoin", "rank": 85},
                {"symbol": "BLUR", "name": "Blur", "rank": 90},
                {"symbol": "ENS", "name": "Ethereum Name Service", "rank": 95},
                {"symbol": "LOOKS", "name": "LooksRare", "rank": 200},
                {"symbol": "X2Y2", "name": "X2Y2", "rank": 250},
                {"symbol": "RARE", "name": "SuperRare", "rank": 220},
                {"symbol": "JPEG", "name": "JPEG'd", "rank": 230},
                {"symbol": "BEND", "name": "BendDAO", "rank": 240},
                {"symbol": "SUDO", "name": "sudoswap", "rank": 260},
                {"symbol": "NFT", "name": "APENFT", "rank": 150},
                {"symbol": "MAGIC", "name": "Magic", "rank": 105},
                {"symbol": "PENGU", "name": "Pudgy Penguins", "rank": 45}
            ]
        },
        "RWA (Real World Assets)": {
            "description": "Tokens representing real-world assets on blockchain",
            "risk_level": "Medium",
            "assets": [
                {"symbol": "ONDO", "name": "Ondo Finance", "rank": 47},
                {"symbol": "CFG", "name": "Centrifuge", "rank": 150},
                {"symbol": "MPL", "name": "Maple", "rank": 180},
                {"symbol": "CPOOL", "name": "Clearpool", "rank": 200}
            ]
        },
        "Gaming/Metaverse": {
            "description": "Tokens for blockchain gaming and virtual worlds",
            "risk_level": "High",
            "assets": [
                {"symbol": "SAND", "name": "The Sandbox", "rank": 57},
                {"symbol": "AXS", "name": "Axie Infinity", "rank": 62},
                {"symbol": "MANA", "name": "Decentraland", "rank": 64},
                {"symbol": "GALA", "name": "Gala", "rank": 56},
                {"symbol": "ENJ", "name": "Enjin Coin", "rank": 93},
                {"symbol": "ILV", "name": "Illuvium", "rank": 105},
                {"symbol": "BEAM", "name": "Beam", "rank": 78},
                {"symbol": "RONIN", "name": "Ronin", "rank": 43}
            ]
        },
        "Infrastructure": {
            "description": "Core blockchain infrastructure and oracle services",
            "risk_level": "Medium",
            "assets": [
                {"symbol": "GRT", "name": "The Graph", "rank": 44},
                {"symbol": "FIL", "name": "Filecoin", "rank": 31},
                {"symbol": "AR", "name": "Arweave", "rank": 39},
                {"symbol": "STX", "name": "Stacks", "rank": 37},
                {"symbol": "PYTH", "name": "Pyth Network", "rank": 51},
                {"symbol": "API3", "name": "API3", "rank": 120}
            ]
        },
        "Exchange Tokens": {
            "description": "Native tokens of cryptocurrency exchanges",
            "risk_level": "Medium",
            "assets": [
                {"symbol": "OKB", "name": "OKB", "rank": 17},
                {"symbol": "CRO", "name": "Cronos", "rank": 49},
                {"symbol": "LEO", "name": "UNUS SED LEO", "rank": 23},
                {"symbol": "KCS", "name": "KuCoin Token", "rank": 73},
                {"symbol": "GT", "name": "Gate Token", "rank": 89},
                {"symbol": "HT", "name": "Huobi Token", "rank": 110}
            ]
        },
        "Privacy": {
            "description": "Cryptocurrencies focused on transaction privacy",
            "risk_level": "High",
            "assets": [
                {"symbol": "XMR", "name": "Monero", "rank": 41},
                {"symbol": "ZEC", "name": "Zcash", "rank": 87},
                {"symbol": "DASH", "name": "Dash", "rank": 94}
            ]
        },
        "Staking/Liquid Staking": {
            "description": "Tokens representing staked assets",
            "risk_level": "Low-Medium",
            "assets": [
                {"symbol": "STETH", "name": "Lido Staked ETH", "rank": 8},
                {"symbol": "RETH", "name": "Rocket Pool ETH", "rank": 60},
                {"symbol": "CBETH", "name": "Coinbase Staked ETH", "rank": 55},
                {"symbol": "RPL", "name": "Rocket Pool", "rank": 115}
            ]
        }
    }
    
    @classmethod
    def get_all_assets(cls) -> List[Dict]:
        """Get flat list of all crypto assets"""
        all_assets = []
        for sector, data in cls.SECTORS.items():
            for asset in data["assets"]:
                asset_copy = asset.copy()
                asset_copy["sector"] = sector
                asset_copy["risk_level"] = data["risk_level"]
                all_assets.append(asset_copy)
        return sorted(all_assets, key=lambda x: x["rank"])
    
    @classmethod
    def get_assets_by_sector(cls, sector: str) -> List[Dict]:
        """Get assets for a specific sector"""
        if sector in cls.SECTORS:
            return cls.SECTORS[sector]["assets"]
        return []
    
    @classmethod
    def get_symbols_by_risk(cls, max_risk: str) -> List[str]:
        """Get symbols filtered by maximum risk level"""
        risk_order = ["Low", "Low-Medium", "Medium", "Medium-High", "High", "Very High", "Extreme"]
        max_idx = risk_order.index(max_risk) if max_risk in risk_order else len(risk_order)
        
        symbols = []
        for sector, data in cls.SECTORS.items():
            sector_risk = data["risk_level"]
            if sector_risk in risk_order:
                if risk_order.index(sector_risk) <= max_idx:
                    symbols.extend([a["symbol"] for a in data["assets"]])
        return symbols
    
    @classmethod
    def get_volatility_class(cls, symbol: str) -> str:
        """Get volatility classification for a symbol (used for adaptive thresholds)"""
        for sector, data in cls.SECTORS.items():
            for asset in data["assets"]:
                if asset["symbol"] == symbol:
                    risk = data["risk_level"]
                    if risk == "Extreme":
                        return "extreme"
                    elif risk == "Very High":
                        return "very_high"
                    elif risk in ["High", "Medium-High"]:
                        return "high"
                    else:
                        return "standard"
        return "standard"
    
    @classmethod
    def get_unique_symbols(cls) -> List[str]:
        """Get deduplicated list of all crypto symbols"""
        symbols = set()
        for sector, data in cls.SECTORS.items():
            for asset in data["assets"]:
                symbols.add(asset["symbol"])
        return sorted(list(symbols))
    
    @classmethod
    def get_sector_for_symbol(cls, symbol: str) -> Optional[str]:
        """Get the primary sector for a given symbol"""
        for sector, data in cls.SECTORS.items():
            for asset in data["assets"]:
                if asset["symbol"] == symbol:
                    return sector
        return None


class VolatilityConfig:
    """Volatility-aware training configuration for different asset classes"""
    
    THRESHOLDS = {
        "extreme": {
            "description": "Meme coins, AI agents - extremely volatile",
            "price_move_thresholds": [8.0, 10.0, 12.0, 15.0],
            "prediction_horizons": [3, 5, 7],
            "min_class_balance": 0.05,
            "max_class_balance": 0.95
        },
        "very_high": {
            "description": "AI tokens, NFT projects - very high volatility",
            "price_move_thresholds": [6.0, 8.0, 10.0],
            "prediction_horizons": [5, 7, 10],
            "min_class_balance": 0.06,
            "max_class_balance": 0.94
        },
        "high": {
            "description": "DeFi, Gaming - high volatility",
            "price_move_thresholds": [5.0, 6.0, 8.0],
            "prediction_horizons": [5, 7, 10],
            "min_class_balance": 0.07,
            "max_class_balance": 0.93
        },
        "standard": {
            "description": "Large-cap, infrastructure - standard volatility",
            "price_move_thresholds": [4.0, 5.0, 6.0],
            "prediction_horizons": [5, 7, 10],
            "min_class_balance": 0.08,
            "max_class_balance": 0.92
        }
    }
    
    @classmethod
    def get_config_for_symbol(cls, symbol: str) -> dict:
        """Get volatility-aware training config for a symbol"""
        volatility_class = CryptoAssets.get_volatility_class(symbol)
        return cls.THRESHOLDS.get(volatility_class, cls.THRESHOLDS["standard"])
    
    @classmethod
    def get_thresholds(cls, volatility_class: str) -> List[float]:
        """Get price movement thresholds for a volatility class"""
        config = cls.THRESHOLDS.get(volatility_class, cls.THRESHOLDS["standard"])
        return config["price_move_thresholds"]
    
    @classmethod
    def get_horizons(cls, volatility_class: str) -> List[int]:
        """Get prediction horizons for a volatility class"""
        config = cls.THRESHOLDS.get(volatility_class, cls.THRESHOLDS["standard"])
        return config["prediction_horizons"]


class StockAssets:
    """Complete stock market sectors, industries, and indices"""
    
    SECTORS = {
        "Technology": {
            "description": "Technology companies including software, hardware, and semiconductors",
            "risk_level": "Medium-High",
            "industries": {
                "Software": ["MSFT", "ORCL", "CRM", "ADBE", "NOW", "INTU", "SNOW"],
                "Semiconductors": ["NVDA", "AMD", "INTC", "AVGO", "QCOM", "MU", "TXN", "AMAT"],
                "Hardware": ["AAPL", "DELL", "HPQ", "HPE"],
                "Cloud/Internet": ["GOOGL", "AMZN", "META", "NFLX", "UBER", "ABNB", "SHOP"],
                "Cybersecurity": ["CRWD", "PANW", "FTNT", "ZS", "OKTA"]
            }
        },
        "Healthcare": {
            "description": "Healthcare, pharmaceuticals, and biotechnology",
            "risk_level": "Medium",
            "industries": {
                "Pharmaceuticals": ["JNJ", "PFE", "MRK", "LLY", "ABBV", "BMY"],
                "Biotechnology": ["AMGN", "GILD", "BIIB", "REGN", "VRTX", "MRNA"],
                "Medical Devices": ["MDT", "ABT", "SYK", "ISRG", "BSX", "EW"],
                "Healthcare Services": ["UNH", "CVS", "CI", "HUM", "ELV"],
                "Healthcare REITs": ["WELL", "VTR", "PEAK"]
            }
        },
        "Financial Services": {
            "description": "Banks, insurance, and financial services",
            "risk_level": "Medium",
            "industries": {
                "Banks": ["JPM", "BAC", "WFC", "C", "GS", "MS", "USB"],
                "Insurance": ["BRK.B", "PRU", "MET", "AIG", "ALL", "TRV"],
                "Investment Services": ["BLK", "SCHW", "SPGI", "ICE", "CME"],
                "Credit Cards/Payments": ["V", "MA", "AXP", "PYPL", "SQ"],
                "FinTech": ["COIN", "HOOD", "SOFI", "AFRM"]
            }
        },
        "Consumer Discretionary": {
            "description": "Non-essential consumer goods and services",
            "risk_level": "Medium-High",
            "industries": {
                "Retail": ["AMZN", "HD", "LOW", "TGT", "COST", "TJX", "ROSS"],
                "Automotive": ["TSLA", "F", "GM", "RIVN", "LCID"],
                "Restaurants": ["MCD", "SBUX", "CMG", "YUM", "DRI"],
                "Hotels/Leisure": ["MAR", "HLT", "LVS", "WYNN", "CCL", "RCL"],
                "Apparel": ["NKE", "LULU", "TJX", "GPS", "VFC"]
            }
        },
        "Consumer Staples": {
            "description": "Essential consumer products",
            "risk_level": "Low",
            "industries": {
                "Food & Beverage": ["KO", "PEP", "MDLZ", "GIS", "K", "HSY"],
                "Household Products": ["PG", "CL", "KMB", "CLX", "CHD"],
                "Retail Staples": ["WMT", "COST", "KR", "WBA", "DG"],
                "Tobacco": ["PM", "MO", "BTI"]
            }
        },
        "Energy": {
            "description": "Oil, gas, and energy companies",
            "risk_level": "Medium-High",
            "industries": {
                "Oil & Gas Integrated": ["XOM", "CVX", "COP", "OXY", "EOG"],
                "Oil & Gas E&P": ["PXD", "DVN", "FANG", "MRO"],
                "Oil Services": ["SLB", "HAL", "BKR"],
                "Refining": ["VLO", "PSX", "MPC"],
                "Clean Energy": ["NEE", "ENPH", "SEDG", "FSLR", "RUN"]
            }
        },
        "Industrials": {
            "description": "Industrial and manufacturing companies",
            "risk_level": "Medium",
            "industries": {
                "Aerospace & Defense": ["BA", "LMT", "RTX", "NOC", "GD", "HII"],
                "Industrial Conglomerates": ["GE", "HON", "MMM", "EMR"],
                "Transportation": ["UPS", "FDX", "UNP", "CSX", "NSC"],
                "Airlines": ["DAL", "UAL", "AAL", "LUV", "JBLU"],
                "Machinery": ["CAT", "DE", "CMI", "ITW", "ETN"]
            }
        },
        "Materials": {
            "description": "Raw materials and chemicals",
            "risk_level": "Medium",
            "industries": {
                "Chemicals": ["LIN", "APD", "SHW", "ECL", "DD"],
                "Metals & Mining": ["FCX", "NEM", "NUE", "STLD"],
                "Construction Materials": ["VMC", "MLM", "CX"],
                "Packaging": ["AMCR", "IP", "PKG", "BALL"]
            }
        },
        "Utilities": {
            "description": "Electric, gas, and water utilities",
            "risk_level": "Low",
            "industries": {
                "Electric Utilities": ["NEE", "DUK", "SO", "D", "AEP", "EXC"],
                "Multi-Utilities": ["WEC", "ES", "AEE", "CMS"],
                "Water Utilities": ["AWK", "WTR", "WTRG"],
                "Renewable Energy": ["NEE", "AES", "BEP"]
            }
        },
        "Real Estate": {
            "description": "Real estate investment trusts (REITs)",
            "risk_level": "Medium",
            "industries": {
                "Retail REITs": ["SPG", "O", "NNN", "REG"],
                "Industrial REITs": ["PLD", "DRE", "EGP"],
                "Office REITs": ["BXP", "VNO", "SLG"],
                "Residential REITs": ["EQR", "AVB", "ESS", "MAA"],
                "Data Center REITs": ["EQIX", "DLR", "AMT", "CCI"],
                "Healthcare REITs": ["WELL", "VTR", "PEAK"]
            }
        },
        "Communication Services": {
            "description": "Media, entertainment, and telecommunications",
            "risk_level": "Medium",
            "industries": {
                "Telecom": ["VZ", "T", "TMUS"],
                "Media & Entertainment": ["DIS", "NFLX", "WBD", "PARA", "FOX"],
                "Interactive Media": ["GOOGL", "META", "SNAP", "PINS", "RBLX"],
                "Gaming": ["EA", "TTWO", "ATVI"]
            }
        }
    }
    
    INDICES = {
        "Major US Indices": {
            "SPY": {"name": "S&P 500 ETF", "description": "Tracks 500 largest US companies"},
            "QQQ": {"name": "Nasdaq 100 ETF", "description": "Tracks 100 largest Nasdaq stocks"},
            "DIA": {"name": "Dow Jones ETF", "description": "Tracks 30 blue-chip companies"},
            "IWM": {"name": "Russell 2000 ETF", "description": "Tracks small-cap stocks"},
            "VTI": {"name": "Total Stock Market ETF", "description": "Tracks entire US market"}
        },
        "Sector ETFs": {
            "XLK": {"name": "Technology Select", "sector": "Technology"},
            "XLF": {"name": "Financial Select", "sector": "Financial Services"},
            "XLV": {"name": "Health Care Select", "sector": "Healthcare"},
            "XLE": {"name": "Energy Select", "sector": "Energy"},
            "XLI": {"name": "Industrial Select", "sector": "Industrials"},
            "XLY": {"name": "Consumer Discretionary", "sector": "Consumer Discretionary"},
            "XLP": {"name": "Consumer Staples", "sector": "Consumer Staples"},
            "XLU": {"name": "Utilities Select", "sector": "Utilities"},
            "XLRE": {"name": "Real Estate Select", "sector": "Real Estate"},
            "XLB": {"name": "Materials Select", "sector": "Materials"},
            "XLC": {"name": "Communication Services", "sector": "Communication Services"}
        },
        "Thematic ETFs": {
            "ARKK": {"name": "ARK Innovation", "theme": "Disruptive Innovation"},
            "SOXX": {"name": "Semiconductor ETF", "theme": "Semiconductors"},
            "BOTZ": {"name": "Robotics & AI ETF", "theme": "AI/Robotics"},
            "ICLN": {"name": "Clean Energy ETF", "theme": "Clean Energy"},
            "HACK": {"name": "Cybersecurity ETF", "theme": "Cybersecurity"},
            "FINX": {"name": "FinTech ETF", "theme": "Financial Technology"},
            "IBB": {"name": "Biotech ETF", "theme": "Biotechnology"},
            "TAN": {"name": "Solar ETF", "theme": "Solar Energy"}
        },
        "International": {
            "EFA": {"name": "EAFE ETF", "description": "Developed markets ex-US"},
            "EEM": {"name": "Emerging Markets ETF", "description": "Emerging market stocks"},
            "VEU": {"name": "All-World ex-US", "description": "Global stocks ex-US"},
            "FXI": {"name": "China Large-Cap", "description": "Chinese stocks"},
            "EWJ": {"name": "Japan ETF", "description": "Japanese stocks"}
        },
        "Bond ETFs": {
            "BND": {"name": "Total Bond Market", "description": "US investment-grade bonds"},
            "TLT": {"name": "20+ Year Treasury", "description": "Long-term treasuries"},
            "HYG": {"name": "High Yield Corporate", "description": "High yield bonds"},
            "LQD": {"name": "Investment Grade Corporate", "description": "Corporate bonds"}
        },
        "Commodity ETFs": {
            "GLD": {"name": "Gold ETF", "commodity": "Gold"},
            "SLV": {"name": "Silver ETF", "commodity": "Silver"},
            "USO": {"name": "Oil ETF", "commodity": "Crude Oil"},
            "UNG": {"name": "Natural Gas ETF", "commodity": "Natural Gas"}
        },
        "Leveraged/Inverse": {
            "TQQQ": {"name": "3x Nasdaq Bull", "leverage": "3x Long QQQ"},
            "SQQQ": {"name": "3x Nasdaq Bear", "leverage": "3x Short QQQ"},
            "SPXL": {"name": "3x S&P Bull", "leverage": "3x Long SPY"},
            "SPXS": {"name": "3x S&P Bear", "leverage": "3x Short SPY"},
            "UVXY": {"name": "1.5x VIX Short-Term", "leverage": "1.5x Long VIX"}
        }
    }
    
    @classmethod
    def get_all_stocks(cls) -> List[str]:
        """Get flat list of all stock symbols"""
        symbols = set()
        for sector, data in cls.SECTORS.items():
            for industry, stocks in data["industries"].items():
                symbols.update(stocks)
        return sorted(list(symbols))
    
    @classmethod
    def get_stocks_by_sector(cls, sector: str) -> List[str]:
        """Get all stocks in a specific sector"""
        if sector in cls.SECTORS:
            stocks = []
            for industry, stock_list in cls.SECTORS[sector]["industries"].items():
                stocks.extend(stock_list)
            return stocks
        return []
    
    @classmethod
    def get_all_indices(cls) -> Dict[str, Dict]:
        """Get all ETF indices"""
        all_indices = {}
        for category, indices in cls.INDICES.items():
            for symbol, data in indices.items():
                data_copy = data.copy()
                data_copy["category"] = category
                all_indices[symbol] = data_copy
        return all_indices
    
    @classmethod
    def get_sector_etf(cls, sector: str) -> Optional[str]:
        """Get the ETF symbol for a sector"""
        sector_map = {
            "Technology": "XLK",
            "Financial Services": "XLF",
            "Healthcare": "XLV",
            "Energy": "XLE",
            "Industrials": "XLI",
            "Consumer Discretionary": "XLY",
            "Consumer Staples": "XLP",
            "Utilities": "XLU",
            "Real Estate": "XLRE",
            "Materials": "XLB",
            "Communication Services": "XLC"
        }
        return sector_map.get(sector)


class AssetRecommendationEngine:
    """Generates asset recommendations based on user risk profile"""
    
    @staticmethod
    def get_crypto_recommendations(risk_level: str, max_count: int = 20) -> List[Dict]:
        """Get crypto recommendations based on risk level"""
        risk_sector_mapping = {
            "conservative": ["Stablecoins", "Staking/Liquid Staking"],
            "moderate": ["Layer 1 (L1)", "Stablecoins", "Infrastructure", "Exchange Tokens"],
            "growth": ["Layer 1 (L1)", "Layer 2 (L2)", "DeFi", "AI Tokens", "NFT Projects"],
            "aggressive": ["Layer 1 (L1)", "Layer 2 (L2)", "DeFi", "AI Tokens", "NFT Projects", "Gaming/Metaverse", "RWA (Real World Assets)"],
            "speculative": list(CryptoAssets.SECTORS.keys())
        }
        
        allowed_sectors = risk_sector_mapping.get(risk_level, ["Layer 1 (L1)"])
        
        recommendations = []
        for sector in allowed_sectors:
            if sector in CryptoAssets.SECTORS:
                for asset in CryptoAssets.SECTORS[sector]["assets"]:
                    asset_copy = asset.copy()
                    asset_copy["sector"] = sector
                    recommendations.append(asset_copy)
        
        recommendations.sort(key=lambda x: x["rank"])
        return recommendations[:max_count]
    
    @staticmethod
    def get_stock_recommendations(risk_level: str, max_per_sector: int = 5) -> Dict[str, List[str]]:
        """Get stock recommendations based on risk level"""
        risk_sector_mapping = {
            "conservative": ["Consumer Staples", "Utilities", "Healthcare"],
            "moderate": ["Technology", "Healthcare", "Financial Services", "Consumer Staples"],
            "growth": ["Technology", "Healthcare", "Consumer Discretionary", "Communication Services"],
            "aggressive": ["Technology", "Consumer Discretionary", "Energy", "Financial Services"],
            "speculative": list(StockAssets.SECTORS.keys())
        }
        
        allowed_sectors = risk_sector_mapping.get(risk_level, ["Technology"])
        
        recommendations = {}
        for sector in allowed_sectors:
            if sector in StockAssets.SECTORS:
                stocks = StockAssets.get_stocks_by_sector(sector)
                recommendations[sector] = stocks[:max_per_sector]
        
        return recommendations
    
    @staticmethod
    def get_etf_recommendations(risk_level: str) -> List[str]:
        """Get ETF recommendations based on risk level"""
        base_etfs = ["SPY", "QQQ", "VTI"]
        
        risk_etf_mapping = {
            "conservative": ["BND", "XLP", "XLU", "DIA", "GLD"],
            "moderate": ["XLK", "XLV", "XLF", "IWM", "EFA"],
            "growth": ["XLK", "SOXX", "ARKK", "XLY", "QQQ"],
            "aggressive": ["TQQQ", "SOXX", "ARKK", "IBB", "XLE"],
            "speculative": ["TQQQ", "SOXL", "UVXY", "ARKK", "FNGU"]
        }
        
        return base_etfs + risk_etf_mapping.get(risk_level, [])
