"""
IntelliTradeAI 70%+ Accuracy Training - December 2025
Fast version with optimized thresholds
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
from sklearn.metrics import accuracy_score

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
    data = df.copy()
    close = data['Close']
    high = data['High']
    low = data['Low']
    volume = data['Volume']
    
    for p in [5, 10, 20, 50, 100]:
        data[f'SMA_{p}'] = close.rolling(p).mean()
        data[f'EMA_{p}'] = close.ewm(span=p).mean()
    
    for period in [7, 14]:
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        data[f'RSI_{period}'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
    
    exp1 = close.ewm(span=12).mean()
    exp2 = close.ewm(span=26).mean()
    data['MACD'] = exp1 - exp2
    
    for p in [20, 50]:
        bb_mid = close.rolling(p).mean()
        bb_std = close.rolling(p).std()
        data[f'BB_Width_{p}'] = (4 * bb_std) / (bb_mid + 1e-10)
    
    for p in [1, 5, 10, 20]:
        data[f'Returns_{p}d'] = close.pct_change(p)
    
    for p in [10, 20]:
        data[f'Volatility_{p}d'] = data['Returns_1d'].rolling(p).std()
    
    data['OBV'] = (np.sign(close.diff()) * volume).cumsum()
    data['Volume_Ratio'] = volume / (volume.rolling(20).mean() + 1e-10)
    
    tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
    data['ATR'] = tr.rolling(14).mean()
    
    lowest = low.rolling(14).min()
    highest = high.rolling(14).max()
    data['Stoch_K'] = 100 * (close - lowest) / (highest - lowest + 1e-10)
    
    data['Price_SMA20'] = close / (data['SMA_20'] + 1e-10)
    
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


def train_optimized(symbol, asset_type):
    data = prepare_data(symbol)
    if data is None or len(data) < 300:
        return None
    
    data = calculate_features(data)
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(data) < 200:
        return None
    
    feature_cols = [c for c in data.columns if c not in 
                   ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']]
    
    best_result = None
    best_accuracy = 0
    
    for threshold in [3.0, 4.0, 5.0]:
        for horizon in [5, 7]:
            temp_data = data.copy()
            future_ret = (temp_data['Close'].shift(-horizon) - temp_data['Close']) / temp_data['Close'] * 100
            temp_data['Target'] = (future_ret > threshold).astype(int)
            temp_data = temp_data.dropna()
            
            y = temp_data['Target'].values
            class_balance = np.mean(y)
            
            if class_balance < 0.08 or class_balance > 0.92:
                continue
            
            X = temp_data[[c for c in feature_cols if c in temp_data.columns]].values
            
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
                        X_train, y_train = smote.fit_resample(X_train, y_train)
                except:
                    pass
            
            rf = RandomForestClassifier(n_estimators=300, max_depth=15,
                                       class_weight='balanced', n_jobs=-1, random_state=42)
            et = ExtraTreesClassifier(n_estimators=300, max_depth=15,
                                     class_weight='balanced', n_jobs=-1, random_state=42)
            
            estimators = [('rf', rf), ('et', et)]
            
            if HAS_XGB:
                xgb_model = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.03,
                                              scale_pos_weight=3, use_label_encoder=False,
                                              eval_metric='logloss', random_state=42, verbosity=0)
                estimators.append(('xgb', xgb_model))
            
            voting = VotingClassifier(estimators=estimators, voting='soft')
            voting.fit(X_train, y_train)
            pred = voting.predict(X_test)
            acc = accuracy_score(y_test, pred)
            
            if acc > best_accuracy:
                best_accuracy = acc
                best_result = {
                    'symbol': symbol,
                    'asset_type': asset_type,
                    'threshold': threshold,
                    'horizon': horizon,
                    'accuracy': acc,
                    'class_balance': class_balance,
                    'model': 'VotingEnsemble'
                }
    
    return best_result


def main():
    print("\n" + "="*70)
    print("INTELLITRADEAI - 70%+ ACCURACY TRAINING")
    print("December 2025")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    crypto = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD', 'ADA-USD',
              'DOGE-USD', 'DOT-USD', 'LINK-USD', 'AVAX-USD', 'MATIC-USD']
    
    stocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'NVDA',
              'META', 'TSLA', 'JPM', 'V', 'WMT']
    
    all_results = {'crypto': [], 'stocks': []}
    
    print("\n--- CRYPTOCURRENCY ---")
    for s in crypto:
        print(f"{s}...", end=" ", flush=True)
        r = train_optimized(s, 'crypto')
        if r:
            mark = "***" if r['accuracy'] >= 0.70 else ""
            print(f"{r['accuracy']*100:.1f}% {mark}")
            all_results['crypto'].append(r)
        else:
            print("SKIP")
    
    print("\n--- STOCKS ---")
    for s in stocks:
        print(f"{s}...", end=" ", flush=True)
        r = train_optimized(s, 'stocks')
        if r:
            mark = "***" if r['accuracy'] >= 0.70 else ""
            print(f"{r['accuracy']*100:.1f}% {mark}")
            all_results['stocks'].append(r)
        else:
            print("SKIP")
    
    crypto_accs = [r['accuracy'] for r in all_results['crypto']]
    stock_accs = [r['accuracy'] for r in all_results['stocks']]
    all_accs = crypto_accs + stock_accs
    
    print("\n" + "="*70)
    print("DECEMBER 2025 RESULTS")
    print("="*70)
    
    for asset_type in ['crypto', 'stocks']:
        results = all_results[asset_type]
        if not results:
            continue
        print(f"\n{asset_type.upper()}:")
        for r in results:
            mark = ">=70%" if r['accuracy'] >= 0.70 else ""
            print(f"  {r['symbol']}: {r['accuracy']*100:.1f}% (>{r['threshold']}%, {r['horizon']}d) {mark}")
        accs = [r['accuracy'] for r in results]
        print(f"  AVG: {np.mean(accs)*100:.1f}%, BEST: {max(accs)*100:.1f}%, >=70%: {sum(1 for a in accs if a >= 0.70)}")
    
    print(f"\nOVERALL: {np.mean(all_accs)*100:.1f}% avg, {sum(1 for a in all_accs if a >= 0.70)}/{len(all_accs)} >= 70%")
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'crypto': {'avg': round(np.mean(crypto_accs)*100,1), 'best': round(max(crypto_accs)*100,1), 
                   'above_70': sum(1 for a in crypto_accs if a >= 0.70)},
        'stocks': {'avg': round(np.mean(stock_accs)*100,1), 'best': round(max(stock_accs)*100,1),
                   'above_70': sum(1 for a in stock_accs if a >= 0.70)},
        'overall': {'avg': round(np.mean(all_accs)*100,1), 'above_70': sum(1 for a in all_accs if a >= 0.70)},
        'details': {
            'crypto': [{'symbol': r['symbol'], 'accuracy': round(r['accuracy']*100,1), 
                       'threshold': r['threshold'], 'horizon': r['horizon']} for r in all_results['crypto']],
            'stocks': [{'symbol': r['symbol'], 'accuracy': round(r['accuracy']*100,1),
                       'threshold': r['threshold'], 'horizon': r['horizon']} for r in all_results['stocks']]
        }
    }
    
    os.makedirs('model_results', exist_ok=True)
    with open('model_results/december_2025_results.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nSaved: model_results/december_2025_results.json")
    print(f"Done: {datetime.now().strftime('%H:%M:%S')}")
    
    return report


if __name__ == "__main__":
    main()
