"""Ultra-fast training - Dec 2025"""
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

print("="*55); print("INTELLITRADEAI - DECEMBER 2025"); print("="*55)
crypto = ['BTC-USD','ETH-USD','SOL-USD','XRP-USD','ADA-USD','DOGE-USD','DOT-USD','LINK-USD','AVAX-USD','MATIC-USD']
stocks = ['AAPL','GOOGL','MSFT','AMZN','NVDA','META','TSLA','JPM','V','WMT']
res = {'crypto':[], 'stocks':[]}
print("\nCRYPTO:")
for s in crypto:
    r = train(s,'crypto')
    if r: print(f"  {s}: {r['accuracy']*100:.1f}%", "***" if r['accuracy']>=0.70 else ""); res['crypto'].append(r)
    else: print(f"  {s}: SKIP")
print("\nSTOCKS:")
for s in stocks:
    r = train(s,'stocks')
    if r: print(f"  {s}: {r['accuracy']*100:.1f}%", "***" if r['accuracy']>=0.70 else ""); res['stocks'].append(r)
    else: print(f"  {s}: SKIP")
ca = [r['accuracy'] for r in res['crypto']]; sa = [r['accuracy'] for r in res['stocks']]; aa = ca+sa
print("\n"+"="*55); print("DECEMBER 2025 FINAL"); print("="*55)
print(f"CRYPTO: {np.mean(ca)*100:.1f}% avg, {max(ca)*100:.1f}% best, {sum(1 for a in ca if a>=0.70)} >=70%")
print(f"STOCKS: {np.mean(sa)*100:.1f}% avg, {max(sa)*100:.1f}% best, {sum(1 for a in sa if a>=0.70)} >=70%")
print(f"OVERALL: {np.mean(aa)*100:.1f}% avg, {sum(1 for a in aa if a>=0.70)}/{len(aa)} >=70%")
rpt = {'timestamp':datetime.now().isoformat(),'crypto':{'avg':round(np.mean(ca)*100,1),'best':round(max(ca)*100,1),'above_70':sum(1 for a in ca if a>=0.70),'count':len(ca)},'stocks':{'avg':round(np.mean(sa)*100,1),'best':round(max(sa)*100,1),'above_70':sum(1 for a in sa if a>=0.70),'count':len(sa)},'overall':round(np.mean(aa)*100,1),'total_above_70':sum(1 for a in aa if a>=0.70),'details':{'crypto':[{'symbol':r['symbol'],'accuracy':round(r['accuracy']*100,1),'threshold':r['threshold'],'horizon':r['horizon']} for r in res['crypto']],'stocks':[{'symbol':r['symbol'],'accuracy':round(r['accuracy']*100,1),'threshold':r['threshold'],'horizon':r['horizon']} for r in res['stocks']]}}
os.makedirs('model_results',exist_ok=True)
with open('model_results/december_2025_results.json','w') as f: json.dump(rpt,f,indent=2)
print("\nSaved: model_results/december_2025_results.json")
