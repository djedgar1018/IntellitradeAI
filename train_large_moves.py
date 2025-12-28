"""
Training for Larger/Significant Moves Only
Hypothesis: Predicting significant moves (>1-2%) is more reliable than noise-dominated small moves
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, f1_score, classification_report

try:
    import xgboost as xgb
    HAS_XGB = True
except:
    HAS_XGB = False


def calculate_features(df):
    """Calculate technical indicators"""
    data = df.copy()
    close = data['Close']
    high = data['High']
    low = data['Low']
    volume = data['Volume']
    
    for p in [5, 10, 20, 50]:
        data[f'SMA_{p}'] = close.rolling(p).mean()
        data[f'EMA_{p}'] = close.ewm(span=p).mean()
    
    delta = close.diff()
    for period in [7, 14, 21]:
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        data[f'RSI_{period}'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
    
    for fast, slow in [(12, 26), (5, 35)]:
        exp1 = close.ewm(span=fast).mean()
        exp2 = close.ewm(span=slow).mean()
        data[f'MACD_{fast}_{slow}'] = exp1 - exp2
    
    for p in [10, 20]:
        bb_mid = close.rolling(p).mean()
        bb_std = close.rolling(p).std()
        data[f'BB_Width_{p}'] = (2 * bb_std) / (bb_mid + 1e-10)
        data[f'BB_Pos_{p}'] = (close - (bb_mid - 2*bb_std)) / (4*bb_std + 1e-10)
    
    for p in [1, 2, 3, 5, 10, 20]:
        data[f'Returns_{p}d'] = close.pct_change(p)
    
    for p in [5, 10, 20]:
        data[f'Volatility_{p}d'] = data['Returns_1d'].rolling(p).std()
    
    data['Volume_Ratio'] = volume / (volume.rolling(20).mean() + 1e-10)
    data['ATR'] = (high - low).rolling(14).mean()
    
    lowest = low.rolling(14).min()
    highest = high.rolling(14).max()
    data['Stoch_K'] = 100 * (close - lowest) / (highest - lowest + 1e-10)
    
    data['ADX'] = calculate_adx(high, low, close)
    data['CCI'] = (close - close.rolling(20).mean()) / (0.015 * close.rolling(20).std() + 1e-10)
    
    data['Trend_Strength'] = abs(data['Returns_20d']) / (data['Volatility_20d'] + 1e-10)
    
    return data


def calculate_adx(high, low, close, period=14):
    tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / (atr + 1e-10))
    minus_di = 100 * (minus_dm.rolling(period).mean() / (atr + 1e-10))
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    return dx.rolling(period).mean()


def create_targets(df, threshold_pct=1.0, horizon=5):
    """Create targets for significant moves"""
    data = df.copy()
    future_return = (data['Close'].shift(-horizon) - data['Close']) / data['Close'] * 100
    
    data['Target_Up'] = (future_return > threshold_pct).astype(int)
    data['Target_Down'] = (future_return < -threshold_pct).astype(int)
    data['Target_Significant'] = ((future_return > threshold_pct) | (future_return < -threshold_pct)).astype(int)
    data['Target_Direction'] = (future_return > 0).astype(int)
    data['Future_Return'] = future_return
    
    return data


def prepare_data(symbol, days=365*3):
    try:
        import yfinance as yf
        data = yf.download(symbol, start=datetime.now() - timedelta(days=days), 
                          end=datetime.now(), progress=False)
        if data.empty:
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data
    except:
        return None


def evaluate_strategies(symbol):
    """Evaluate different prediction strategies"""
    print(f"\n{'='*60}")
    print(f"MULTI-STRATEGY EVALUATION: {symbol}")
    print(f"{'='*60}")
    
    data = prepare_data(symbol)
    if data is None or len(data) < 300:
        print(f"Insufficient data")
        return None
    
    print(f"Data points: {len(data)}")
    
    strategies = [
        {'name': '1-Day Direction', 'horizon': 1, 'threshold': 0, 'target': 'direction'},
        {'name': '3-Day Direction', 'horizon': 3, 'threshold': 0, 'target': 'direction'},
        {'name': '5-Day Direction', 'horizon': 5, 'threshold': 0, 'target': 'direction'},
        {'name': '5-Day >1% Up', 'horizon': 5, 'threshold': 1.0, 'target': 'up'},
        {'name': '5-Day >2% Up', 'horizon': 5, 'threshold': 2.0, 'target': 'up'},
        {'name': '5-Day >3% Up', 'horizon': 5, 'threshold': 3.0, 'target': 'up'},
        {'name': '10-Day Direction', 'horizon': 10, 'threshold': 0, 'target': 'direction'},
        {'name': '10-Day >2% Up', 'horizon': 10, 'threshold': 2.0, 'target': 'up'},
    ]
    
    results = []
    
    for strat in strategies:
        df = calculate_features(data.copy())
        df = create_targets(df, threshold_pct=strat['threshold'], horizon=strat['horizon'])
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        
        if strat['target'] == 'direction':
            target_col = 'Target_Direction'
        elif strat['target'] == 'up':
            target_col = 'Target_Up'
        else:
            target_col = 'Target_Direction'
        
        feature_cols = [c for c in df.columns if c not in 
                       ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close',
                        'Target_Up', 'Target_Down', 'Target_Significant', 
                        'Target_Direction', 'Future_Return']]
        
        X = df[feature_cols].values
        y = df[target_col].values
        
        class_balance = np.mean(y)
        if class_balance < 0.1 or class_balance > 0.9:
            print(f"  {strat['name']}: Skipped (imbalanced: {class_balance:.1%})")
            continue
        
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        best_acc = 0
        best_model = None
        
        models = [
            ('RF', RandomForestClassifier(n_estimators=200, max_depth=10, n_jobs=-1, random_state=42)),
            ('GB', GradientBoostingClassifier(n_estimators=150, max_depth=4, random_state=42)),
            ('ET', ExtraTreesClassifier(n_estimators=200, max_depth=10, n_jobs=-1, random_state=42)),
        ]
        
        if HAS_XGB:
            models.append(('XGB', xgb.XGBClassifier(n_estimators=200, max_depth=5, 
                          use_label_encoder=False, eval_metric='logloss', 
                          random_state=42, verbosity=0)))
        
        for name, model in models:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            acc = accuracy_score(y_test, pred)
            if acc > best_acc:
                best_acc = acc
                best_model = name
        
        results.append({
            'strategy': strat['name'],
            'accuracy': best_acc,
            'model': best_model,
            'class_balance': class_balance
        })
        print(f"  {strat['name']}: {best_acc*100:.1f}% ({best_model}) [balance: {class_balance:.1%}]")
    
    return results


def main():
    print("\n" + "="*70)
    print("ALTERNATIVE PREDICTION STRATEGIES EVALUATION")
    print("="*70)
    print("Testing: Different horizons and move thresholds")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    crypto = ['BTC-USD', 'ETH-USD', 'SOL-USD']
    stocks = ['AAPL', 'MSFT', 'NVDA']
    
    print("\n--- CRYPTOCURRENCY ---")
    crypto_results = {}
    for s in crypto:
        r = evaluate_strategies(s)
        if r:
            crypto_results[s] = r
    
    print("\n--- STOCKS ---")
    stock_results = {}
    for s in stocks:
        r = evaluate_strategies(s)
        if r:
            stock_results[s] = r
    
    print("\n" + "="*70)
    print("BEST STRATEGIES BY ASSET")
    print("="*70)
    
    for asset_type, results in [('CRYPTO', crypto_results), ('STOCKS', stock_results)]:
        print(f"\n{asset_type}:")
        for symbol, strats in results.items():
            if strats:
                best = max(strats, key=lambda x: x['accuracy'])
                print(f"  {symbol}: {best['strategy']} = {best['accuracy']*100:.1f}%")
    
    print("\n" + "="*70)
    print("AVERAGE BY STRATEGY ACROSS ALL ASSETS")
    print("="*70)
    
    strategy_accs = {}
    for results in [crypto_results, stock_results]:
        for symbol, strats in results.items():
            for s in strats:
                name = s['strategy']
                if name not in strategy_accs:
                    strategy_accs[name] = []
                strategy_accs[name].append(s['accuracy'])
    
    for name, accs in sorted(strategy_accs.items(), key=lambda x: np.mean(x[1]), reverse=True):
        print(f"  {name}: {np.mean(accs)*100:.1f}% avg")
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
