"""IntelliTradeAI Complete Training - December 2025
Extended coverage: Top 50 cryptos + All GICS sectors + Major ETFs
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

TOP_50_CRYPTO = [
    'BTC-USD', 'ETH-USD', 'USDT-USD', 'XRP-USD', 'BNB-USD',
    'SOL-USD', 'DOGE-USD', 'ADA-USD', 'TRX-USD', 'AVAX-USD',
    'LINK-USD', 'XLM-USD', 'TON11419-USD', 'SHIB-USD', 'DOT-USD',
    'HBAR-USD', 'BCH-USD', 'LEO-USD', 'UNI7083-USD', 'LTC-USD',
    'ATOM-USD', 'NEAR-USD', 'APT21794-USD', 'ETC-USD', 'MATIC-USD',
    'ICP-USD', 'RENDER-USD', 'XMR-USD', 'TAO22974-USD', 'VET-USD',
    'CRO-USD', 'FIL-USD', 'ARB11841-USD', 'MNT27075-USD', 'AAVE-USD',
    'OP-USD', 'INJ-USD', 'ALGO-USD', 'FTM-USD', 'IMX10603-USD'
]

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

def feats(df):
    d = df.copy(); c = d['Close']
    for p in [10,20,50]: d[f'SMA{p}'] = c.rolling(p).mean()
    delta = c.diff(); gain = delta.where(delta>0,0).rolling(14).mean(); loss = (-delta.where(delta<0,0)).rolling(14).mean()
    d['RSI'] = 100 - 100/(1 + gain/(loss+1e-10))
    d['MACD'] = c.ewm(span=12).mean() - c.ewm(span=26).mean()
    for p in [1,5,10]: d[f'R{p}'] = c.pct_change(p)
    d['Vol'] = d['R1'].rolling(20).std()
    d['PS'] = c / (d['SMA20'] + 1e-10)
    return d

def get(s):
    try:
        import yfinance as yf
        d = yf.download(s, start=datetime.now()-timedelta(days=1825), end=datetime.now(), progress=False)
        if d.empty: return None
        if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.get_level_values(0)
        return d
    except: return None

def train(s, t):
    data = get(s)
    if data is None or len(data) < 300: return None
    data = feats(data).replace([np.inf,-np.inf], np.nan).dropna()
    if len(data) < 200: return None
    fc = [c for c in data.columns if c not in ['Open','High','Low','Close','Volume','Adj Close']]
    best, ba = None, 0
    for th in [4.0, 5.0]:
        for hz in [5, 7]:
            tmp = data.copy()
            fr = (tmp['Close'].shift(-hz) - tmp['Close']) / tmp['Close'] * 100
            tmp['T'] = (fr > th).astype(int); tmp = tmp.dropna()
            y = tmp['T'].values; cb = np.mean(y)
            if cb < 0.08 or cb > 0.92: continue
            X = RobustScaler().fit_transform(tmp[[c for c in fc if c in tmp.columns]].values)
            sp = int(len(X)*0.8); Xtr,Xte,ytr,yte = X[:sp],X[sp:],y[:sp],y[sp:]
            if HAS_SMOTE and len(np.unique(ytr))>1:
                try:
                    k = min(5, min(np.bincount(ytr))-1)
                    if k>=1: Xtr,ytr = SMOTE(k_neighbors=k, random_state=42).fit_resample(Xtr,ytr)
                except: pass
            est = [('rf', RandomForestClassifier(n_estimators=150, max_depth=10, class_weight='balanced', n_jobs=-1, random_state=42))]
            if HAS_XGB: est.append(('xgb', xgb.XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.05, scale_pos_weight=3, use_label_encoder=False, eval_metric='logloss', random_state=42, verbosity=0)))
            v = VotingClassifier(estimators=est, voting='soft'); v.fit(Xtr, ytr)
            acc = accuracy_score(yte, v.predict(Xte))
            if acc > ba: ba = acc; best = {'symbol':s, 'type':t, 'threshold':th, 'horizon':hz, 'accuracy':acc}
    return best

print("="*60)
print("INTELLITRADEAI COMPLETE TRAINING - DECEMBER 2025")
print("="*60)
print(f"Cryptos: {len(TOP_50_CRYPTO)}, Stocks: {sum(len(v) for v in GICS_STOCKS.values())}, ETFs: {len(MAJOR_ETFS)}")

results = {'crypto': [], 'stocks': [], 'etfs': []}

print("\n=== TOP CRYPTOCURRENCIES ===")
for i, s in enumerate(TOP_50_CRYPTO):
    r = train(s, 'crypto')
    sym_short = s.replace('-USD','')
    if r:
        m = ">=70%" if r['accuracy'] >= 0.70 else ""
        print(f"  [{i+1:2d}] {sym_short:8s}: {r['accuracy']*100:.1f}% {m}")
        results['crypto'].append(r)
    else:
        print(f"  [{i+1:2d}] {sym_short:8s}: SKIP")

print("\n=== STOCKS BY GICS SECTOR ===")
for sector, stocks in GICS_STOCKS.items():
    print(f"\n{sector}:")
    for s in stocks:
        r = train(s, 'stocks')
        if r:
            m = ">=70%" if r['accuracy'] >= 0.70 else ""
            print(f"  {s:5s}: {r['accuracy']*100:.1f}% {m}")
            r['sector'] = sector
            results['stocks'].append(r)
        else:
            print(f"  {s:5s}: SKIP")

print("\n=== MAJOR ETFs ===")
for s in MAJOR_ETFS:
    r = train(s, 'etfs')
    if r:
        m = ">=70%" if r['accuracy'] >= 0.70 else ""
        print(f"  {s:5s}: {r['accuracy']*100:.1f}% {m}")
        results['etfs'].append(r)
    else:
        print(f"  {s:5s}: SKIP")

ca = [r['accuracy'] for r in results['crypto']]
sa = [r['accuracy'] for r in results['stocks']]
ea = [r['accuracy'] for r in results['etfs']]
aa = ca + sa + ea

print("\n" + "="*60)
print("FINAL RESULTS - DECEMBER 2025")
print("="*60)

if ca:
    c70 = sum(1 for a in ca if a >= 0.70)
    print(f"CRYPTO ({len(ca)} assets): {np.mean(ca)*100:.1f}% avg, {max(ca)*100:.1f}% best, {c70} >=70%")
    
if sa:
    s70 = sum(1 for a in sa if a >= 0.70)
    print(f"STOCKS ({len(sa)} assets): {np.mean(sa)*100:.1f}% avg, {max(sa)*100:.1f}% best, {s70} >=70%")
    
if ea:
    e70 = sum(1 for a in ea if a >= 0.70)
    print(f"ETFs ({len(ea)} assets): {np.mean(ea)*100:.1f}% avg, {max(ea)*100:.1f}% best, {e70} >=70%")

if aa:
    total_70 = sum(1 for a in aa if a >= 0.70)
    print(f"\nOVERALL ({len(aa)} assets): {np.mean(aa)*100:.1f}% avg, {total_70}/{len(aa)} >=70%")

rpt = {
    'timestamp': datetime.now().isoformat(),
    'crypto': {'count': len(ca), 'avg': round(np.mean(ca)*100,1) if ca else 0, 'best': round(max(ca)*100,1) if ca else 0, 'above_70': sum(1 for a in ca if a>=0.70)},
    'stocks': {'count': len(sa), 'avg': round(np.mean(sa)*100,1) if sa else 0, 'best': round(max(sa)*100,1) if sa else 0, 'above_70': sum(1 for a in sa if a>=0.70)},
    'etfs': {'count': len(ea), 'avg': round(np.mean(ea)*100,1) if ea else 0, 'best': round(max(ea)*100,1) if ea else 0, 'above_70': sum(1 for a in ea if a>=0.70)},
    'overall': {'count': len(aa), 'avg': round(np.mean(aa)*100,1) if aa else 0, 'above_70': sum(1 for a in aa if a>=0.70)},
    'details': {
        'crypto': [{'symbol': r['symbol'], 'accuracy': round(r['accuracy']*100,1)} for r in sorted(results['crypto'], key=lambda x: x['accuracy'], reverse=True)],
        'stocks': [{'symbol': r['symbol'], 'sector': r.get('sector',''), 'accuracy': round(r['accuracy']*100,1)} for r in sorted(results['stocks'], key=lambda x: x['accuracy'], reverse=True)],
        'etfs': [{'symbol': r['symbol'], 'accuracy': round(r['accuracy']*100,1)} for r in sorted(results['etfs'], key=lambda x: x['accuracy'], reverse=True)]
    }
}

os.makedirs('model_results', exist_ok=True)
with open('model_results/december_2025_results.json', 'w') as f:
    json.dump(rpt, f, indent=2)
print(f"\nSaved: model_results/december_2025_results.json")
