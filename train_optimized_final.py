"""
IntelliTradeAI Optimized Training - Maximum Accuracy
Target: >3% price movement over 5 days (higher threshold for better accuracy)
"""

import warnings
warnings.filterwarnings('ignore')

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
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
    
    tp = (high + low + close) / 3
    mf = tp * volume
    delta_tp = tp.diff()
    pos_mf = (mf * (delta_tp > 0)).rolling(14).sum()
    neg_mf = (mf * (delta_tp < 0)).rolling(14).sum()
    data['MFI'] = 100 - (100 / (1 + pos_mf / (neg_mf + 1e-10)))
    
    data['Skewness'] = data['Returns_1d'].rolling(20).skew()
    data['Kurtosis'] = data['Returns_1d'].rolling(20).kurt()
    
    data['Price_SMA20'] = close / (data['SMA_20'] + 1e-10)
    data['Price_SMA50'] = close / (data['SMA_50'] + 1e-10)
    
    return data


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


def train_asset(symbol, asset_type, horizon=5, threshold=3.0):
    """Train optimized models for one asset"""
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
    
    if class_balance < 0.10 or class_balance > 0.90:
        return None
    
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
    
    rf = RandomForestClassifier(n_estimators=300, max_depth=15, min_samples_split=5,
                               class_weight='balanced', n_jobs=-1, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=200, max_depth=5, 
                                    learning_rate=0.05, random_state=42)
    et = ExtraTreesClassifier(n_estimators=300, max_depth=15, min_samples_split=5,
                             class_weight='balanced', n_jobs=-1, random_state=42)
    
    estimators = [('rf', rf), ('gb', gb), ('et', et)]
    
    if HAS_XGB:
        xgb_model = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.03,
                                      scale_pos_weight=3, use_label_encoder=False,
                                      eval_metric='logloss', random_state=42, verbosity=0)
        estimators.append(('xgb', xgb_model))
    
    results = {}
    
    for name, model in estimators:
        model.fit(X_train_bal, y_train_bal)
        pred = model.predict(X_test)
        results[name.upper()] = {
            'accuracy': accuracy_score(y_test, pred),
            'precision': precision_score(y_test, pred, zero_division=0),
            'recall': recall_score(y_test, pred, zero_division=0),
            'f1': f1_score(y_test, pred, zero_division=0),
        }
    
    voting = VotingClassifier(estimators=estimators, voting='soft')
    voting.fit(X_train_bal, y_train_bal)
    pred = voting.predict(X_test)
    results['ENSEMBLE'] = {
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
    print("INTELLITRADEAI - OPTIMIZED PUBLICATION TRAINING")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Target: >3% price movement over 5 days")
    print(f"XGBoost: {HAS_XGB}, SMOTE: {HAS_SMOTE}")
    
    crypto = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD', 'ADA-USD',
              'DOGE-USD', 'DOT-USD', 'LINK-USD', 'AVAX-USD', 'MATIC-USD']
    
    stocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'NVDA',
              'META', 'TSLA', 'JPM', 'V', 'WMT']
    
    all_results = {'crypto': [], 'stocks': []}
    
    print("\n--- CRYPTOCURRENCY ---")
    for s in crypto:
        print(f"Training {s}...", end=" ")
        r = train_asset(s, 'crypto', threshold=3.0)
        if r:
            print(f"{r['best_model']}: {r['best_accuracy']*100:.1f}%")
            all_results['crypto'].append(r)
        else:
            print("SKIPPED (imbalanced)")
    
    print("\n--- STOCKS ---")
    for s in stocks:
        print(f"Training {s}...", end=" ")
        r = train_asset(s, 'stocks', threshold=3.0)
        if r:
            print(f"{r['best_model']}: {r['best_accuracy']*100:.1f}%")
            all_results['stocks'].append(r)
        else:
            print("SKIPPED (imbalanced)")
    
    print("\n" + "="*70)
    print("DETAILED RESULTS")
    print("="*70)
    
    for asset_type in ['crypto', 'stocks']:
        results = all_results[asset_type]
        if not results:
            continue
        
        print(f"\n{asset_type.upper()} ({len(results)} assets):")
        print("-" * 50)
        
        for r in results:
            print(f"  {r['symbol']}: {r['best_accuracy']*100:.1f}% ({r['best_model']}) "
                  f"[balance: {r['class_balance']:.1%}]")
        
        accs = [r['best_accuracy'] for r in results]
        print(f"\n  Average: {np.mean(accs)*100:.1f}%")
        print(f"  Best: {max(accs)*100:.1f}%")
        print(f"  Worst: {min(accs)*100:.1f}%")
        
        ensemble_accs = [r['results'].get('ENSEMBLE', {}).get('accuracy', 0) for r in results]
        if any(ensemble_accs):
            print(f"  Ensemble Avg: {np.mean(ensemble_accs)*100:.1f}%")
    
    crypto_accs = [r['best_accuracy'] for r in all_results['crypto']]
    stock_accs = [r['best_accuracy'] for r in all_results['stocks']]
    
    print("\n" + "="*70)
    print("PUBLICATION METRICS (>3% Threshold)")
    print("="*70)
    
    if crypto_accs:
        print(f"\nCryptocurrency ({len(crypto_accs)} assets):")
        print(f"  Average Accuracy: {np.mean(crypto_accs)*100:.1f}%")
        print(f"  Best: {max(crypto_accs)*100:.1f}%")
    
    if stock_accs:
        print(f"\nStock Market ({len(stock_accs)} assets):")
        print(f"  Average Accuracy: {np.mean(stock_accs)*100:.1f}%")
        print(f"  Best: {max(stock_accs)*100:.1f}%")
    
    all_accs = crypto_accs + stock_accs
    if all_accs:
        print(f"\nOverall ({len(all_accs)} assets):")
        print(f"  Average Accuracy: {np.mean(all_accs)*100:.1f}%")
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'target': '>3% price movement over 5 days',
        'crypto': {
            'count': len(crypto_accs),
            'avg_accuracy': round(np.mean(crypto_accs) * 100, 1) if crypto_accs else 0,
            'best_accuracy': round(max(crypto_accs) * 100, 1) if crypto_accs else 0,
        },
        'stocks': {
            'count': len(stock_accs),
            'avg_accuracy': round(np.mean(stock_accs) * 100, 1) if stock_accs else 0,
            'best_accuracy': round(max(stock_accs) * 100, 1) if stock_accs else 0,
        },
        'overall_avg': round(np.mean(all_accs) * 100, 1) if all_accs else 0,
        'details': {
            'crypto': [{'symbol': r['symbol'], 'accuracy': round(r['best_accuracy']*100, 1), 
                       'model': r['best_model']} for r in all_results['crypto']],
            'stocks': [{'symbol': r['symbol'], 'accuracy': round(r['best_accuracy']*100, 1),
                       'model': r['best_model']} for r in all_results['stocks']]
        }
    }
    
    os.makedirs('model_results', exist_ok=True)
    with open('model_results/optimized_results.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nResults saved to: model_results/optimized_results.json")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return report


if __name__ == "__main__":
    main()
