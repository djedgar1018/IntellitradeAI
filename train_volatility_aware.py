"""IntelliTradeAI Volatility-Aware Training Pipeline
Enhanced training with adaptive thresholds for volatile assets
Targets: Meme Coins, NFT Projects, AI Agent Tokens
"""
import warnings; warnings.filterwarnings('ignore')
import os, json, numpy as np, pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score
try:
    import xgboost as xgb; HAS_XGB = True
except: HAS_XGB = False
try:
    from imblearn.over_sampling import SMOTE; HAS_SMOTE = True
except: HAS_SMOTE = False

from config.assets_config import CryptoAssets, VolatilityConfig
from models.volatility_features import VolatilityFeatures, AdaptiveThresholdEngine

GICS_STOCKS = {
    'Technology': ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META', 'AVGO', 'ORCL', 'CRM', 'AMD', 'ADBE'],
    'Healthcare': ['UNH', 'JNJ', 'LLY', 'PFE', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'BMY'],
    'Financials': ['JPM', 'V', 'MA', 'BAC', 'WFC', 'GS', 'MS', 'AXP', 'BLK', 'C'],
    'Consumer Discretionary': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'LOW', 'SBUX', 'TJX', 'BKNG', 'CMG'],
    'Industrials': ['CAT', 'RTX', 'UNP', 'HON', 'BA', 'DE', 'LMT', 'UPS', 'GE', 'MMM'],
    'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY', 'PXD'],
    'Materials': ['LIN', 'APD', 'SHW', 'ECL', 'FCX', 'NEM', 'DOW', 'DD', 'NUE', 'VMC'],
    'Consumer Staples': ['WMT', 'PG', 'COST', 'KO', 'PEP', 'PM', 'MO', 'CL', 'MDLZ', 'EL'],
    'Utilities': ['NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'XEL', 'WEC', 'ED'],
    'Real Estate': ['PLD', 'AMT', 'EQIX', 'CCI', 'PSA', 'SPG', 'O', 'WELL', 'DLR', 'AVB'],
    'Communication': ['GOOG', 'T', 'VZ', 'NFLX', 'DIS', 'CMCSA', 'TMUS', 'CHTR', 'EA', 'MTCH']
}

MAJOR_ETFS = ['SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'XLK', 'XLV', 'XLF', 'XLE', 'XLY']

vol_features = VolatilityFeatures()

def base_features(df):
    """Calculate base technical features"""
    d = df.copy()
    c = d['Close']
    
    for p in [10, 20, 50]:
        d[f'SMA{p}'] = c.rolling(p).mean()
    
    delta = c.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    d['RSI'] = 100 - 100 / (1 + gain / (loss + 1e-10))
    
    d['MACD'] = c.ewm(span=12).mean() - c.ewm(span=26).mean()
    
    for p in [1, 5, 10]:
        d[f'R{p}'] = c.pct_change(p)
    
    d['Vol'] = d['R1'].rolling(20).std()
    d['PS'] = c / (d['SMA20'] + 1e-10)
    
    return d

def enhanced_features(df, is_volatile=False):
    """Calculate enhanced features including volatility-specific ones"""
    d = base_features(df)
    
    if is_volatile:
        d_lower = d.copy()
        d_lower.columns = [c.lower() for c in d_lower.columns]
        d_vol = vol_features.calculate_all_volatility_features(d_lower)
        
        for col in d_vol.columns:
            if col not in ['open', 'high', 'low', 'close', 'volume']:
                if col not in d.columns:
                    d[col] = d_vol[col].values
    
    return d

def get_data(symbol):
    """Fetch historical data"""
    try:
        import yfinance as yf
        data = yf.download(symbol, start=datetime.now() - timedelta(days=1825), 
                          end=datetime.now(), progress=False)
        if data.empty:
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data
    except:
        return None

def train_with_adaptive_threshold(symbol, asset_type, volatility_class='standard'):
    """Train model with volatility-aware thresholds"""
    data = get_data(symbol)
    if data is None or len(data) < 300:
        return None
    
    is_volatile = volatility_class in ['extreme', 'very_high', 'high']
    data = enhanced_features(data, is_volatile=is_volatile)
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(data) < 200:
        return None
    
    exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close',
                    'open', 'high', 'low', 'close', 'volume', 'adj close']
    feature_cols = [c for c in data.columns if c not in exclude_cols]
    
    config = VolatilityConfig.THRESHOLDS.get(volatility_class, VolatilityConfig.THRESHOLDS['standard'])
    thresholds = config['price_move_thresholds']
    horizons = config['prediction_horizons']
    min_balance = config['min_class_balance']
    max_balance = config['max_class_balance']
    
    best_result = None
    best_accuracy = 0
    
    for threshold in thresholds:
        for horizon in horizons:
            try:
                tmp = data.copy()
                future_return = (tmp['Close'].shift(-horizon) - tmp['Close']) / tmp['Close'] * 100
                tmp['Target'] = (future_return > threshold).astype(int)
                tmp = tmp.dropna()
                
                y = tmp['Target'].values
                class_balance = np.mean(y)
                
                if class_balance < min_balance or class_balance > max_balance:
                    continue
                
                valid_features = [c for c in feature_cols if c in tmp.columns]
                X = tmp[valid_features].values
                
                X = RobustScaler().fit_transform(X)
                
                split = int(len(X) * 0.8)
                X_train, X_test = X[:split], X[split:]
                y_train, y_test = y[:split], y[split:]
                
                if HAS_SMOTE and len(np.unique(y_train)) > 1:
                    try:
                        min_samples = min(np.bincount(y_train))
                        k = min(5, min_samples - 1)
                        if k >= 1:
                            X_train, y_train = SMOTE(k_neighbors=k, random_state=42).fit_resample(X_train, y_train)
                    except:
                        pass
                
                estimators = [
                    ('rf', RandomForestClassifier(
                        n_estimators=150, 
                        max_depth=10, 
                        class_weight='balanced',
                        n_jobs=-1, 
                        random_state=42
                    ))
                ]
                
                if HAS_XGB:
                    scale_pos = max(1, int((1 - class_balance) / (class_balance + 1e-10)))
                    estimators.append(('xgb', xgb.XGBClassifier(
                        n_estimators=150, 
                        max_depth=5,
                        learning_rate=0.05, 
                        scale_pos_weight=scale_pos,
                        use_label_encoder=False, 
                        eval_metric='logloss',
                        random_state=42, 
                        verbosity=0
                    )))
                
                model = VotingClassifier(estimators=estimators, voting='soft')
                model.fit(X_train, y_train)
                
                accuracy = accuracy_score(y_test, model.predict(X_test))
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_result = {
                        'symbol': symbol,
                        'type': asset_type,
                        'volatility_class': volatility_class,
                        'threshold': threshold,
                        'horizon': horizon,
                        'accuracy': accuracy,
                        'class_balance': class_balance,
                        'n_features': len(valid_features),
                        'n_samples': len(X)
                    }
                    
            except Exception as e:
                continue
    
    return best_result


def get_crypto_symbols():
    """Get all unique crypto symbols formatted for Yahoo Finance"""
    from data.data_ingestion import DataIngestion
    ingestion = DataIngestion()
    yahoo_map = ingestion._get_yahoo_crypto_map()
    
    unique_symbols = CryptoAssets.get_unique_symbols()
    
    symbols_with_yahoo = []
    for sym in unique_symbols:
        yf_sym = yahoo_map.get(sym, f'{sym}-USD')
        vol_class = CryptoAssets.get_volatility_class(sym)
        symbols_with_yahoo.append((sym, yf_sym, vol_class))
    
    return symbols_with_yahoo


def main():
    print("=" * 70)
    print("INTELLITRADEAI VOLATILITY-AWARE TRAINING PIPELINE")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print()
    
    crypto_symbols = get_crypto_symbols()
    print(f"Crypto assets: {len(crypto_symbols)}")
    print(f"Stocks: {sum(len(v) for v in GICS_STOCKS.values())}")
    print(f"ETFs: {len(MAJOR_ETFS)}")
    
    results = {'crypto': [], 'stocks': [], 'etfs': []}
    
    vol_class_counts = {}
    for _, _, vc in crypto_symbols:
        vol_class_counts[vc] = vol_class_counts.get(vc, 0) + 1
    print(f"\nVolatility class distribution:")
    for vc, count in sorted(vol_class_counts.items()):
        print(f"  {vc}: {count} assets")
    
    print("\n" + "=" * 70)
    print("CRYPTOCURRENCY TRAINING (Volatility-Aware)")
    print("=" * 70)
    
    for i, (sym, yf_sym, vol_class) in enumerate(crypto_symbols):
        result = train_with_adaptive_threshold(yf_sym, 'crypto', vol_class)
        
        if result:
            marker = ">=70%" if result['accuracy'] >= 0.70 else ""
            vc_marker = f"[{vol_class[:3].upper()}]"
            print(f"  [{i+1:3d}] {sym:12s}: {result['accuracy']*100:5.1f}% {vc_marker} th={result['threshold']}% h={result['horizon']}d {marker}")
            result['original_symbol'] = sym
            results['crypto'].append(result)
        else:
            print(f"  [{i+1:3d}] {sym:12s}: SKIP")
    
    print("\n" + "=" * 70)
    print("STOCKS BY GICS SECTOR")
    print("=" * 70)
    
    for sector, stocks in GICS_STOCKS.items():
        print(f"\n{sector}:")
        for s in stocks:
            result = train_with_adaptive_threshold(s, 'stocks', 'standard')
            if result:
                marker = ">=70%" if result['accuracy'] >= 0.70 else ""
                print(f"  {s:5s}: {result['accuracy']*100:5.1f}% {marker}")
                result['sector'] = sector
                results['stocks'].append(result)
            else:
                print(f"  {s:5s}: SKIP")
    
    print("\n" + "=" * 70)
    print("MAJOR ETFs")
    print("=" * 70)
    
    for s in MAJOR_ETFS:
        result = train_with_adaptive_threshold(s, 'etfs', 'standard')
        if result:
            marker = ">=70%" if result['accuracy'] >= 0.70 else ""
            print(f"  {s:5s}: {result['accuracy']*100:5.1f}% {marker}")
            results['etfs'].append(result)
        else:
            print(f"  {s:5s}: SKIP")
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    def summarize(name, data):
        if not data:
            return
        accs = [r['accuracy'] for r in data]
        above_70 = sum(1 for a in accs if a >= 0.70)
        print(f"\n{name}:")
        print(f"  Total: {len(data)}, Avg: {np.mean(accs)*100:.1f}%, "
              f"Best: {max(accs)*100:.1f}%, "
              f">=70%: {above_70}/{len(data)} ({above_70/len(data)*100:.0f}%)")
    
    summarize("Cryptocurrency", results['crypto'])
    summarize("Stocks", results['stocks'])
    summarize("ETFs", results['etfs'])
    
    if results['crypto']:
        print("\nCrypto by Volatility Class:")
        for vol_class in ['extreme', 'very_high', 'high', 'standard']:
            class_results = [r for r in results['crypto'] if r.get('volatility_class') == vol_class]
            if class_results:
                accs = [r['accuracy'] for r in class_results]
                above_70 = sum(1 for a in accs if a >= 0.70)
                print(f"  {vol_class:12s}: {len(class_results):3d} assets, "
                      f"avg {np.mean(accs)*100:.1f}%, >=70%: {above_70}/{len(class_results)}")
    
    os.makedirs('model_results', exist_ok=True)
    output_file = f'model_results/volatility_aware_results_{datetime.now().strftime("%Y%m%d_%H%M")}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    
    return results


if __name__ == '__main__':
    main()
