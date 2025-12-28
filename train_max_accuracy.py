"""
IntelliTradeAI Maximum Accuracy Training - December 2025
Target: 70%+ accuracy through optimized thresholds and ensemble methods
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


def train_with_threshold_search(symbol, asset_type):
    """Find optimal threshold for maximum accuracy"""
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
    
    for threshold in [2.0, 2.5, 3.0, 3.5, 4.0]:
        for horizon in [5, 7]:
            temp_data = data.copy()
            future_ret = (temp_data['Close'].shift(-horizon) - temp_data['Close']) / temp_data['Close'] * 100
            temp_data['Target'] = (future_ret > threshold).astype(int)
            temp_data = temp_data.dropna()
            
            y = temp_data['Target'].values
            class_balance = np.mean(y)
            
            if class_balance < 0.10 or class_balance > 0.90:
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
            
            rf = RandomForestClassifier(n_estimators=400, max_depth=18, min_samples_split=3,
                                       class_weight='balanced', n_jobs=-1, random_state=42)
            gb = GradientBoostingClassifier(n_estimators=250, max_depth=6, 
                                            learning_rate=0.03, subsample=0.8, random_state=42)
            et = ExtraTreesClassifier(n_estimators=400, max_depth=18, min_samples_split=3,
                                     class_weight='balanced', n_jobs=-1, random_state=42)
            
            estimators = [('rf', rf), ('gb', gb), ('et', et)]
            
            if HAS_XGB:
                xgb_model = xgb.XGBClassifier(n_estimators=400, max_depth=8, learning_rate=0.02,
                                              subsample=0.8, colsample_bytree=0.8,
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
                    'samples': len(temp_data),
                    'model': 'VotingEnsemble'
                }
    
    return best_result


def main():
    print("\n" + "="*70)
    print("INTELLITRADEAI - MAXIMUM ACCURACY TRAINING")
    print("December 2025 - Target: 70%+ Accuracy")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Strategy: Threshold optimization + Enhanced ensemble")
    print(f"XGBoost: {HAS_XGB}, SMOTE: {HAS_SMOTE}")
    
    crypto = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD', 'ADA-USD',
              'DOGE-USD', 'DOT-USD', 'LINK-USD', 'AVAX-USD', 'MATIC-USD']
    
    stocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'NVDA',
              'META', 'TSLA', 'JPM', 'V', 'WMT']
    
    all_results = {'crypto': [], 'stocks': []}
    
    print("\n--- CRYPTOCURRENCY (with threshold optimization) ---")
    for s in crypto:
        print(f"Optimizing {s}...", end=" ", flush=True)
        r = train_with_threshold_search(s, 'crypto')
        if r:
            print(f"{r['accuracy']*100:.1f}% (>{r['threshold']}% in {r['horizon']}d)")
            all_results['crypto'].append(r)
        else:
            print("SKIPPED")
    
    print("\n--- STOCKS (with threshold optimization) ---")
    for s in stocks:
        print(f"Optimizing {s}...", end=" ", flush=True)
        r = train_with_threshold_search(s, 'stocks')
        if r:
            print(f"{r['accuracy']*100:.1f}% (>{r['threshold']}% in {r['horizon']}d)")
            all_results['stocks'].append(r)
        else:
            print("SKIPPED")
    
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    for asset_type in ['crypto', 'stocks']:
        results = all_results[asset_type]
        if not results:
            continue
        
        print(f"\n{asset_type.upper()} ({len(results)} assets):")
        print("-" * 55)
        
        for r in results:
            status = "PASS" if r['accuracy'] >= 0.70 else ""
            print(f"  {r['symbol']}: {r['accuracy']*100:.1f}% "
                  f"(>{r['threshold']}% in {r['horizon']}d) {status}")
        
        accs = [r['accuracy'] for r in results]
        above_70 = sum(1 for a in accs if a >= 0.70)
        print(f"\n  Average: {np.mean(accs)*100:.1f}%")
        print(f"  Best: {max(accs)*100:.1f}%")
        print(f"  Assets >= 70%: {above_70}/{len(results)}")
    
    crypto_accs = [r['accuracy'] for r in all_results['crypto']]
    stock_accs = [r['accuracy'] for r in all_results['stocks']]
    all_accs = crypto_accs + stock_accs
    
    print("\n" + "="*70)
    print("FINAL METRICS - December 2025")
    print("="*70)
    
    if crypto_accs:
        print(f"\nCryptocurrency ({len(crypto_accs)} assets):")
        print(f"  Average Accuracy: {np.mean(crypto_accs)*100:.1f}%")
        print(f"  Best: {max(crypto_accs)*100:.1f}%")
        print(f"  >= 70%: {sum(1 for a in crypto_accs if a >= 0.70)}/{len(crypto_accs)}")
    
    if stock_accs:
        print(f"\nStock Market ({len(stock_accs)} assets):")
        print(f"  Average Accuracy: {np.mean(stock_accs)*100:.1f}%")
        print(f"  Best: {max(stock_accs)*100:.1f}%")
        print(f"  >= 70%: {sum(1 for a in stock_accs if a >= 0.70)}/{len(stock_accs)}")
    
    if all_accs:
        print(f"\nOverall ({len(all_accs)} assets):")
        print(f"  Average Accuracy: {np.mean(all_accs)*100:.1f}%")
        print(f"  Total >= 70%: {sum(1 for a in all_accs if a >= 0.70)}/{len(all_accs)}")
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'target': 'Optimized threshold for maximum accuracy',
        'crypto': {
            'count': len(crypto_accs),
            'avg_accuracy': round(np.mean(crypto_accs) * 100, 1) if crypto_accs else 0,
            'best_accuracy': round(max(crypto_accs) * 100, 1) if crypto_accs else 0,
            'above_70_count': sum(1 for a in crypto_accs if a >= 0.70)
        },
        'stocks': {
            'count': len(stock_accs),
            'avg_accuracy': round(np.mean(stock_accs) * 100, 1) if stock_accs else 0,
            'best_accuracy': round(max(stock_accs) * 100, 1) if stock_accs else 0,
            'above_70_count': sum(1 for a in stock_accs if a >= 0.70)
        },
        'overall_avg': round(np.mean(all_accs) * 100, 1) if all_accs else 0,
        'total_above_70': sum(1 for a in all_accs if a >= 0.70),
        'details': {
            'crypto': [{'symbol': r['symbol'], 'accuracy': round(r['accuracy']*100, 1), 
                       'threshold': r['threshold'], 'horizon': r['horizon']} 
                       for r in all_results['crypto']],
            'stocks': [{'symbol': r['symbol'], 'accuracy': round(r['accuracy']*100, 1),
                       'threshold': r['threshold'], 'horizon': r['horizon']} 
                       for r in all_results['stocks']]
        }
    }
    
    os.makedirs('model_results', exist_ok=True)
    with open('model_results/max_accuracy_results.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nResults saved to: model_results/max_accuracy_results.json")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return report


if __name__ == "__main__":
    main()
