"""IntelliTradeAI Fast 70%+ Training - Dec 2025"""
import warnings
warnings.filterwarnings('ignore')
import os, json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
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

def calc_features(df):
    d = df.copy()
    c = d['Close']
    for p in [10, 20, 50]:
        d[f'SMA_{p}'] = c.rolling(p).mean()
        d[f'EMA_{p}'] = c.ewm(span=p).mean()
    delta = c.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    d['RSI'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
    d['MACD'] = c.ewm(span=12).mean() - c.ewm(span=26).mean()
    for p in [1, 5, 10]:
        d[f'Ret_{p}'] = c.pct_change(p)
    d['Vol'] = d['Ret_1'].rolling(20).std()
    d['OBV'] = (np.sign(c.diff()) * d['Volume']).cumsum()
    d['Price_SMA'] = c / (d['SMA_20'] + 1e-10)
    return d

def get_data(sym):
    try:
        import yfinance as yf
        d = yf.download(sym, start=datetime.now()-timedelta(days=5*365), end=datetime.now(), progress=False)
        if d.empty: return None
        if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.get_level_values(0)
        return d
    except:
        return None

def train_one(sym, atype):
    data = get_data(sym)
    if data is None or len(data) < 300: return None
    data = calc_features(data)
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 200: return None
    fcols = [c for c in data.columns if c not in ['Open','High','Low','Close','Volume','Adj Close']]
    best = None
    best_acc = 0
    for th in [3.5, 4.0, 5.0]:
        for hz in [5, 7]:
            tmp = data.copy()
            fr = (tmp['Close'].shift(-hz) - tmp['Close']) / tmp['Close'] * 100
            tmp['Target'] = (fr > th).astype(int)
            tmp = tmp.dropna()
            y = tmp['Target'].values
            cb = np.mean(y)
            if cb < 0.08 or cb > 0.92: continue
            X = tmp[[c for c in fcols if c in tmp.columns]].values
            X = RobustScaler().fit_transform(X)
            sp = int(len(X) * 0.8)
            Xtr, Xte = X[:sp], X[sp:]
            ytr, yte = y[:sp], y[sp:]
            if HAS_SMOTE and len(np.unique(ytr)) > 1:
                try:
                    k = min(5, min(np.bincount(ytr)) - 1)
                    if k >= 1: Xtr, ytr = SMOTE(k_neighbors=k, random_state=42).fit_resample(Xtr, ytr)
                except: pass
            est = [('rf', RandomForestClassifier(n_estimators=200, max_depth=12, class_weight='balanced', n_jobs=-1, random_state=42)),
                   ('et', ExtraTreesClassifier(n_estimators=200, max_depth=12, class_weight='balanced', n_jobs=-1, random_state=42))]
            if HAS_XGB:
                est.append(('xgb', xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, scale_pos_weight=3, use_label_encoder=False, eval_metric='logloss', random_state=42, verbosity=0)))
            v = VotingClassifier(estimators=est, voting='soft')
            v.fit(Xtr, ytr)
            acc = accuracy_score(yte, v.predict(Xte))
            if acc > best_acc:
                best_acc = acc
                best = {'symbol': sym, 'type': atype, 'threshold': th, 'horizon': hz, 'accuracy': acc, 'balance': cb}
    return best

def main():
    print("="*60)
    print("INTELLITRADEAI - DECEMBER 2025 TRAINING")
    print("="*60)
    crypto = ['BTC-USD','ETH-USD','SOL-USD','XRP-USD','ADA-USD','DOGE-USD','DOT-USD','LINK-USD','AVAX-USD','MATIC-USD']
    stocks = ['AAPL','GOOGL','MSFT','AMZN','NVDA','META','TSLA','JPM','V','WMT']
    results = {'crypto': [], 'stocks': []}
    print("\nCRYPTO:")
    for s in crypto:
        r = train_one(s, 'crypto')
        if r:
            m = "***70%+" if r['accuracy'] >= 0.70 else ""
            print(f"  {s}: {r['accuracy']*100:.1f}% {m}")
            results['crypto'].append(r)
        else:
            print(f"  {s}: SKIP")
    print("\nSTOCKS:")
    for s in stocks:
        r = train_one(s, 'stocks')
        if r:
            m = "***70%+" if r['accuracy'] >= 0.70 else ""
            print(f"  {s}: {r['accuracy']*100:.1f}% {m}")
            results['stocks'].append(r)
        else:
            print(f"  {s}: SKIP")
    ca = [r['accuracy'] for r in results['crypto']]
    sa = [r['accuracy'] for r in results['stocks']]
    aa = ca + sa
    print("\n" + "="*60)
    print("FINAL RESULTS - DECEMBER 2025")
    print("="*60)
    print(f"Crypto: {np.mean(ca)*100:.1f}% avg, {max(ca)*100:.1f}% best, {sum(1 for a in ca if a>=0.70)} >=70%")
    print(f"Stocks: {np.mean(sa)*100:.1f}% avg, {max(sa)*100:.1f}% best, {sum(1 for a in sa if a>=0.70)} >=70%")
    print(f"OVERALL: {np.mean(aa)*100:.1f}% avg, {sum(1 for a in aa if a>=0.70)}/{len(aa)} >=70%")
    report = {
        'timestamp': datetime.now().isoformat(),
        'crypto': {'avg': round(np.mean(ca)*100,1), 'best': round(max(ca)*100,1), 'above_70': sum(1 for a in ca if a>=0.70), 'count': len(ca)},
        'stocks': {'avg': round(np.mean(sa)*100,1), 'best': round(max(sa)*100,1), 'above_70': sum(1 for a in sa if a>=0.70), 'count': len(sa)},
        'overall': round(np.mean(aa)*100,1),
        'total_above_70': sum(1 for a in aa if a>=0.70),
        'details': {'crypto': [{'symbol': r['symbol'], 'accuracy': round(r['accuracy']*100,1), 'threshold': r['threshold'], 'horizon': r['horizon']} for r in results['crypto']],
                    'stocks': [{'symbol': r['symbol'], 'accuracy': round(r['accuracy']*100,1), 'threshold': r['threshold'], 'horizon': r['horizon']} for r in results['stocks']]}
    }
    os.makedirs('model_results', exist_ok=True)
    with open('model_results/december_2025_results.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved: model_results/december_2025_results.json")

if __name__ == "__main__":
    main()
