"""
IntelliTradeAI Final Publication Training - Optimized for Speed
Target: >2% price movement over 5 days
"""

import warnings
warnings.filterwarnings('ignore')

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

try:
    import xgboost as xgb
    HAS_XGB = True
except:
    HAS_XGB = False

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except:
    HAS_SMOTE = False


def calculate_features(df):
    """Calculate technical indicators"""
    data = df.copy()
    close = data['Close']
    high = data['High']
    low = data['Low']
    volume = data['Volume']
    
    for p in [5, 10, 20, 50, 100, 200]:
        data[f'SMA_{p}'] = close.rolling(p).mean()
    
    for p in [5, 10, 12, 20, 26, 50]:
        data[f'EMA_{p}'] = close.ewm(span=p).mean()
    
    for period in [7, 14, 21]:
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        data[f'RSI_{period}'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
    
    for fast, slow in [(12, 26), (5, 35)]:
        exp1 = close.ewm(span=fast).mean()
        exp2 = close.ewm(span=slow).mean()
        data[f'MACD_{fast}_{slow}'] = exp1 - exp2
    
    for p in [10, 20, 50]:
        bb_mid = close.rolling(p).mean()
        bb_std = close.rolling(p).std()
        data[f'BB_Width_{p}'] = (4 * bb_std) / (bb_mid + 1e-10)
        data[f'BB_Pos_{p}'] = (close - (bb_mid - 2*bb_std)) / (4*bb_std + 1e-10)
    
    for p in [1, 2, 3, 5, 10, 20]:
        data[f'Returns_{p}d'] = close.pct_change(p)
    
    for p in [5, 10, 20, 50]:
        data[f'Volatility_{p}d'] = data['Returns_1d'].rolling(p).std()
    
    data['OBV'] = (np.sign(close.diff()) * volume).cumsum()
    for p in [5, 10, 20]:
        data[f'Volume_Ratio_{p}'] = volume / (volume.rolling(p).mean() + 1e-10)
    
    for p in [7, 14]:
        tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
        data[f'ATR_{p}'] = tr.rolling(p).mean()
    
    for p in [5, 14]:
        lowest = low.rolling(p).min()
        highest = high.rolling(p).max()
        data[f'Stoch_K_{p}'] = 100 * (close - lowest) / (highest - lowest + 1e-10)
    
    data['CCI'] = (close - close.rolling(20).mean()) / (0.015 * close.rolling(20).std() + 1e-10)
    data['MFI'] = calculate_mfi(high, low, close, volume)
    
    data['Skewness'] = data['Returns_1d'].rolling(20).skew()
    data['Kurtosis'] = data['Returns_1d'].rolling(20).kurt()
    
    return data


def calculate_mfi(high, low, close, volume, period=14):
    tp = (high + low + close) / 3
    mf = tp * volume
    delta_tp = tp.diff()
    pos_mf = (mf * (delta_tp > 0)).rolling(period).sum()
    neg_mf = (mf * (delta_tp < 0)).rolling(period).sum()
    return 100 - (100 / (1 + pos_mf / (neg_mf + 1e-10)))


def prepare_data(symbol, years=5):
    try:
        import yfinance as yf
        data = yf.download(symbol, start=datetime.now() - timedelta(days=years*365),
                          end=datetime.now(), progress=False)
        if data.empty:
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data
    except:
        return None


def train_asset(symbol, asset_type, horizon=5, threshold=2.0):
    """Train models for one asset"""
    data = prepare_data(symbol)
    if data is None or len(data) < 300:
        return None
    
    data = calculate_features(data)
    future_ret = (data['Close'].shift(-horizon) - data['Close']) / data['Close'] * 100
    data['Target'] = (future_ret > threshold).astype(int)
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(data) < 200:
        return None
    
    feature_cols = [c for c in data.columns if c not in 
                   ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 'Target']]
    
    X = data[feature_cols].values
    y = data['Target'].values
    
    class_balance = np.mean(y)
    
    scaler = RobustScaler()
    X = scaler.fit_transform(X)
    
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    if HAS_SMOTE and len(np.unique(y_train)) > 1:
        try:
            k = min(5, min(np.bincount(y_train)) - 1)
            if k >= 1:
                smote = SMOTE(k_neighbors=k, random_state=42)
                X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
            else:
                X_train_bal, y_train_bal = X_train, y_train
        except:
            X_train_bal, y_train_bal = X_train, y_train
    else:
        X_train_bal, y_train_bal = X_train, y_train
    
    results = {}
    
    models = {
        'RF': RandomForestClassifier(n_estimators=250, max_depth=12, 
                                    class_weight='balanced', n_jobs=-1, random_state=42),
        'GB': GradientBoostingClassifier(n_estimators=150, max_depth=4, 
                                         learning_rate=0.08, random_state=42),
        'ET': ExtraTreesClassifier(n_estimators=250, max_depth=12,
                                   class_weight='balanced', n_jobs=-1, random_state=42),
    }
    
    if HAS_XGB:
        models['XGB'] = xgb.XGBClassifier(n_estimators=250, max_depth=5, learning_rate=0.05,
                                          scale_pos_weight=2, use_label_encoder=False,
                                          eval_metric='logloss', random_state=42, verbosity=0)
    
    for name, model in models.items():
        model.fit(X_train_bal, y_train_bal)
        pred = model.predict(X_test)
        results[name] = {
            'accuracy': accuracy_score(y_test, pred),
            'precision': precision_score(y_test, pred, zero_division=0),
            'recall': recall_score(y_test, pred, zero_division=0),
            'f1': f1_score(y_test, pred, zero_division=0),
        }
    
    best = max(results.items(), key=lambda x: x[1]['accuracy'])
    
    return {
        'symbol': symbol,
        'asset_type': asset_type,
        'samples': len(data),
        'class_balance': class_balance,
        'results': results,
        'best_model': best[0],
        'best_accuracy': best[1]['accuracy']
    }


def main():
    print("\n" + "="*70)
    print("INTELLITRADEAI - FINAL PUBLICATION TRAINING")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Target: >2% price movement over 5 days")
    print(f"XGBoost: {HAS_XGB}, SMOTE: {HAS_SMOTE}")
    
    crypto = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD', 'ADA-USD',
              'DOGE-USD', 'DOT-USD', 'LINK-USD', 'AVAX-USD', 'MATIC-USD']
    
    stocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'NVDA',
              'META', 'TSLA', 'JPM', 'V', 'WMT']
    
    all_results = {'crypto': [], 'stocks': []}
    
    print("\n--- CRYPTOCURRENCY ---")
    for s in crypto:
        print(f"Training {s}...", end=" ")
        r = train_asset(s, 'crypto')
        if r:
            print(f"{r['best_model']}: {r['best_accuracy']*100:.1f}%")
            all_results['crypto'].append(r)
        else:
            print("FAILED")
    
    print("\n--- STOCKS ---")
    for s in stocks:
        print(f"Training {s}...", end=" ")
        r = train_asset(s, 'stocks')
        if r:
            print(f"{r['best_model']}: {r['best_accuracy']*100:.1f}%")
            all_results['stocks'].append(r)
        else:
            print("FAILED")
    
    print("\n" + "="*70)
    print("DETAILED RESULTS")
    print("="*70)
    
    for asset_type in ['crypto', 'stocks']:
        results = all_results[asset_type]
        if not results:
            continue
        
        print(f"\n{asset_type.upper()}:")
        print("-" * 50)
        
        for r in results:
            print(f"  {r['symbol']}: {r['best_accuracy']*100:.1f}% ({r['best_model']}) "
                  f"[balance: {r['class_balance']:.1%}]")
        
        accs = [r['best_accuracy'] for r in results]
        print(f"\n  Average: {np.mean(accs)*100:.1f}%")
        print(f"  Best: {max(accs)*100:.1f}%")
        print(f"  Worst: {min(accs)*100:.1f}%")
        
        model_accs = {}
        for r in results:
            for m, metrics in r['results'].items():
                if m not in model_accs:
                    model_accs[m] = []
                model_accs[m].append(metrics['accuracy'])
        
        print("\n  By Model:")
        for m, accs in sorted(model_accs.items(), key=lambda x: np.mean(x[1]), reverse=True):
            print(f"    {m}: {np.mean(accs)*100:.1f}% avg")
    
    crypto_accs = [r['best_accuracy'] for r in all_results['crypto']]
    stock_accs = [r['best_accuracy'] for r in all_results['stocks']]
    
    print("\n" + "="*70)
    print("PUBLICATION METRICS")
    print("="*70)
    print(f"\nCryptocurrency ({len(crypto_accs)} assets):")
    print(f"  Average Accuracy: {np.mean(crypto_accs)*100:.1f}%")
    print(f"  Best: {max(crypto_accs)*100:.1f}%")
    
    print(f"\nStock Market ({len(stock_accs)} assets):")
    print(f"  Average Accuracy: {np.mean(stock_accs)*100:.1f}%")
    print(f"  Best: {max(stock_accs)*100:.1f}%")
    
    all_accs = crypto_accs + stock_accs
    print(f"\nOverall ({len(all_accs)} assets):")
    print(f"  Average Accuracy: {np.mean(all_accs)*100:.1f}%")
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'target': '>2% price movement over 5 days',
        'crypto': {
            'count': len(crypto_accs),
            'avg_accuracy': round(np.mean(crypto_accs) * 100, 1),
            'best_accuracy': round(max(crypto_accs) * 100, 1),
        },
        'stocks': {
            'count': len(stock_accs),
            'avg_accuracy': round(np.mean(stock_accs) * 100, 1),
            'best_accuracy': round(max(stock_accs) * 100, 1),
        },
        'overall_avg': round(np.mean(all_accs) * 100, 1),
        'details': {
            'crypto': [{'symbol': r['symbol'], 'accuracy': round(r['best_accuracy']*100, 1), 
                       'model': r['best_model']} for r in all_results['crypto']],
            'stocks': [{'symbol': r['symbol'], 'accuracy': round(r['best_accuracy']*100, 1),
                       'model': r['best_model']} for r in all_results['stocks']]
        }
    }
    
    os.makedirs('model_results', exist_ok=True)
    with open('model_results/final_publication_results.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nResults saved to: model_results/final_publication_results.json")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return report


if __name__ == "__main__":
    main()
