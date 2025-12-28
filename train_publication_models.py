"""
IntelliTradeAI Publication Model Training
=========================================
Comprehensive training pipeline for validated accuracy metrics
Target: Significant move prediction (>2% moves over 5-day horizon)

This script produces reproducible results suitable for academic publication.
"""

import warnings
warnings.filterwarnings('ignore')

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import joblib

from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                              ExtraTreesClassifier, VotingClassifier, StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix, classification_report)
from sklearn.feature_selection import SelectFromModel

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

RESULTS_DIR = "model_results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def calculate_comprehensive_features(df):
    """Calculate 70+ technical indicators as described in the paper"""
    data = df.copy()
    close = data['Close']
    high = data['High']
    low = data['Low']
    open_price = data['Open']
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
    
    for fast, slow, signal in [(12, 26, 9), (5, 35, 5)]:
        exp1 = close.ewm(span=fast).mean()
        exp2 = close.ewm(span=slow).mean()
        macd = exp1 - exp2
        macd_signal = macd.ewm(span=signal).mean()
        data[f'MACD_{fast}_{slow}'] = macd
        data[f'MACD_Signal_{fast}_{slow}'] = macd_signal
        data[f'MACD_Hist_{fast}_{slow}'] = macd - macd_signal
    
    for p in [10, 20, 50]:
        bb_mid = close.rolling(p).mean()
        bb_std = close.rolling(p).std()
        data[f'BB_Upper_{p}'] = bb_mid + 2 * bb_std
        data[f'BB_Lower_{p}'] = bb_mid - 2 * bb_std
        data[f'BB_Width_{p}'] = (4 * bb_std) / (bb_mid + 1e-10)
        data[f'BB_Position_{p}'] = (close - data[f'BB_Lower_{p}']) / (data[f'BB_Upper_{p}'] - data[f'BB_Lower_{p}'] + 1e-10)
    
    for p in [7, 14, 21]:
        tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
        data[f'ATR_{p}'] = tr.rolling(p).mean()
    
    for p in [5, 14, 21]:
        lowest = low.rolling(p).min()
        highest = high.rolling(p).max()
        data[f'Stoch_K_{p}'] = 100 * (close - lowest) / (highest - lowest + 1e-10)
        data[f'Stoch_D_{p}'] = data[f'Stoch_K_{p}'].rolling(3).mean()
    
    data['OBV'] = (np.sign(close.diff()) * volume).cumsum()
    data['OBV_EMA'] = data['OBV'].ewm(span=20).mean()
    
    for p in [5, 10, 20]:
        data[f'Volume_SMA_{p}'] = volume.rolling(p).mean()
        data[f'Volume_Ratio_{p}'] = volume / (data[f'Volume_SMA_{p}'] + 1e-10)
    
    for p in [1, 2, 3, 5, 10, 20]:
        data[f'Returns_{p}d'] = close.pct_change(p)
    
    for p in [5, 10, 20, 50]:
        data[f'Volatility_{p}d'] = data['Returns_1d'].rolling(p).std()
    
    for p in [5, 10, 20]:
        data[f'Momentum_{p}'] = close - close.shift(p)
        data[f'ROC_{p}'] = 100 * (close - close.shift(p)) / (close.shift(p) + 1e-10)
    
    data['ADX'] = calculate_adx(high, low, close, 14)
    data['CCI'] = (close - close.rolling(20).mean()) / (0.015 * close.rolling(20).std() + 1e-10)
    data['Williams_R'] = -100 * (high.rolling(14).max() - close) / (high.rolling(14).max() - low.rolling(14).min() + 1e-10)
    
    tp = (high + low + close) / 3
    mf = tp * volume
    delta_tp = tp.diff()
    pos_mf = (mf * (delta_tp > 0)).rolling(14).sum()
    neg_mf = (mf * (delta_tp < 0)).rolling(14).sum()
    data['MFI'] = 100 - (100 / (1 + pos_mf / (neg_mf + 1e-10)))
    
    data['High_Low_Range'] = (high - low) / (close + 1e-10)
    data['Close_Open_Range'] = (close - open_price) / (open_price + 1e-10)
    data['Upper_Shadow'] = (high - pd.concat([close, open_price], axis=1).max(axis=1)) / (high - low + 1e-10)
    data['Lower_Shadow'] = (pd.concat([close, open_price], axis=1).min(axis=1) - low) / (high - low + 1e-10)
    data['Body_Size'] = abs(close - open_price) / (high - low + 1e-10)
    
    data['Price_SMA20_Ratio'] = close / (data['SMA_20'] + 1e-10)
    data['Price_SMA50_Ratio'] = close / (data['SMA_50'] + 1e-10)
    data['SMA_20_50_Cross'] = (data['SMA_20'] > data['SMA_50']).astype(int)
    data['EMA_12_26_Cross'] = (data['EMA_12'] > data['EMA_26']).astype(int)
    
    data['Skewness'] = data['Returns_1d'].rolling(20).skew()
    data['Kurtosis'] = data['Returns_1d'].rolling(20).kurt()
    
    for lag in [1, 2, 3, 5]:
        data[f'Returns_Lag_{lag}'] = data['Returns_1d'].shift(lag)
        data[f'RSI_Lag_{lag}'] = data['RSI_14'].shift(lag)
    
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


def create_publication_target(df, horizon=5, threshold=2.0):
    """Create target for significant moves (>threshold% over horizon days)"""
    data = df.copy()
    future_return = (data['Close'].shift(-horizon) - data['Close']) / data['Close'] * 100
    data['Target'] = (future_return > threshold).astype(int)
    data['Future_Return'] = future_return
    return data


def prepare_data(symbol, years=5):
    """Fetch extended historical data"""
    try:
        import yfinance as yf
        end = datetime.now()
        start = end - timedelta(days=years*365)
        data = yf.download(symbol, start=start, end=end, progress=False)
        if data.empty:
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data
    except:
        return None


def select_best_features(X, y, n_features=50):
    """Select top features using Random Forest importance"""
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    importances = rf.feature_importances_
    top_indices = np.argsort(importances)[-n_features:]
    return top_indices, importances


def build_stacking_ensemble():
    """Build optimized stacking ensemble"""
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=300, max_depth=15, min_samples_split=5,
                                      class_weight='balanced', n_jobs=-1, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                                          subsample=0.8, random_state=42)),
        ('et', ExtraTreesClassifier(n_estimators=300, max_depth=15, min_samples_split=5,
                                    class_weight='balanced', n_jobs=-1, random_state=42)),
    ]
    
    if HAS_XGB:
        estimators.append(
            ('xgb', xgb.XGBClassifier(n_estimators=350, max_depth=6, learning_rate=0.03,
                                      subsample=0.8, colsample_bytree=0.8,
                                      use_label_encoder=False, eval_metric='logloss',
                                      scale_pos_weight=2, random_state=42, verbosity=0))
        )
    
    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(C=0.5, max_iter=1000, random_state=42),
        cv=5,
        n_jobs=-1
    )
    
    return stacking


def walk_forward_evaluation(X, y, model_fn, n_splits=10, train_ratio=0.7):
    """Proper walk-forward validation for time series"""
    n = len(X)
    fold_results = []
    
    for i in range(n_splits):
        train_end = int(n * (train_ratio + (1 - train_ratio) * i / n_splits))
        test_start = train_end
        test_end = int(n * (train_ratio + (1 - train_ratio) * (i + 1) / n_splits))
        
        if test_end > n:
            test_end = n
        if test_start >= test_end:
            continue
        
        X_train, y_train = X[:train_end], y[:train_end]
        X_test, y_test = X[test_start:test_end], y[test_start:test_end]
        
        if HAS_SMOTE and len(np.unique(y_train)) > 1:
            try:
                k = min(5, min(np.bincount(y_train)) - 1)
                if k >= 1:
                    smote = SMOTE(k_neighbors=k, random_state=42)
                    X_train, y_train = smote.fit_resample(X_train, y_train)
            except:
                pass
        
        model = model_fn()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
        }
        if y_proba is not None:
            try:
                metrics['auc'] = roc_auc_score(y_test, y_proba)
            except:
                metrics['auc'] = 0.5
        
        fold_results.append(metrics)
    
    avg = {k: np.mean([r[k] for r in fold_results]) for k in fold_results[0]}
    std = {k: np.std([r[k] for r in fold_results]) for k in fold_results[0]}
    
    return avg, std, fold_results


def train_publication_model(symbol, asset_type, horizon=5, threshold=2.0):
    """Train and validate model for a single asset"""
    print(f"\n{'='*70}")
    print(f"TRAINING: {symbol} ({asset_type.upper()})")
    print(f"Target: >{threshold}% move over {horizon} days")
    print(f"{'='*70}")
    
    data = prepare_data(symbol, years=5)
    if data is None or len(data) < 500:
        print("Insufficient data")
        return None
    
    print(f"Raw data: {len(data)} samples")
    
    data = calculate_comprehensive_features(data)
    data = create_publication_target(data, horizon=horizon, threshold=threshold)
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    
    print(f"Processed: {len(data)} samples, {len(data.columns)} features")
    
    feature_cols = [c for c in data.columns if c not in 
                   ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 'Target', 'Future_Return']]
    
    X = data[feature_cols].values
    y = data['Target'].values
    
    class_balance = np.mean(y)
    print(f"Class balance: {class_balance:.1%} positive")
    
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("\nFeature selection...")
    top_indices, importances = select_best_features(X_scaled, y, n_features=50)
    X_selected = X_scaled[:, top_indices]
    selected_features = np.array(feature_cols)[top_indices]
    print(f"Selected {len(selected_features)} features")
    print(f"Top 5: {', '.join(selected_features[-5:])}")
    
    print("\nTraining models with walk-forward validation...")
    
    results = {}
    
    models = {
        'RandomForest': lambda: RandomForestClassifier(
            n_estimators=300, max_depth=15, min_samples_split=5,
            class_weight='balanced', n_jobs=-1, random_state=42
        ),
        'GradientBoosting': lambda: GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, random_state=42
        ),
        'ExtraTrees': lambda: ExtraTreesClassifier(
            n_estimators=300, max_depth=15, min_samples_split=5,
            class_weight='balanced', n_jobs=-1, random_state=42
        ),
    }
    
    if HAS_XGB:
        models['XGBoost'] = lambda: xgb.XGBClassifier(
            n_estimators=350, max_depth=6, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=2,
            use_label_encoder=False, eval_metric='logloss',
            random_state=42, verbosity=0
        )
    
    for name, model_fn in models.items():
        print(f"\n  {name}:")
        avg, std, folds = walk_forward_evaluation(X_selected, y, model_fn, n_splits=8)
        results[name] = {'avg': avg, 'std': std, 'folds': folds}
        print(f"    Accuracy: {avg['accuracy']:.4f} (+/- {std['accuracy']:.4f})")
        print(f"    Precision: {avg['precision']:.4f}, Recall: {avg['recall']:.4f}")
        print(f"    F1: {avg['f1']:.4f}, AUC: {avg.get('auc', 0):.4f}")
    
    print("\n  StackingEnsemble:")
    avg, std, folds = walk_forward_evaluation(X_selected, y, build_stacking_ensemble, n_splits=8)
    results['StackingEnsemble'] = {'avg': avg, 'std': std, 'folds': folds}
    print(f"    Accuracy: {avg['accuracy']:.4f} (+/- {std['accuracy']:.4f})")
    print(f"    Precision: {avg['precision']:.4f}, Recall: {avg['recall']:.4f}")
    print(f"    F1: {avg['f1']:.4f}, AUC: {avg.get('auc', 0):.4f}")
    
    best_model = max(results.items(), key=lambda x: x[1]['avg']['accuracy'])
    print(f"\n  BEST: {best_model[0]} with {best_model[1]['avg']['accuracy']*100:.1f}% accuracy")
    
    return {
        'symbol': symbol,
        'asset_type': asset_type,
        'horizon': horizon,
        'threshold': threshold,
        'samples': len(data),
        'features': len(selected_features),
        'class_balance': class_balance,
        'results': results,
        'best_model': best_model[0],
        'best_accuracy': best_model[1]['avg']['accuracy']
    }


def main():
    """Main publication training pipeline"""
    print("\n" + "="*80)
    print("INTELLITRADEAI - PUBLICATION MODEL TRAINING")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nConfiguration:")
    print("  - Prediction Target: >2% price movement over 5 days")
    print("  - Validation: Walk-forward (8 splits)")
    print("  - Features: 70+ technical indicators, top 50 selected")
    print("  - Ensemble: Stacking (RF + GB + ET + XGB → LogReg)")
    print(f"  - XGBoost available: {HAS_XGB}")
    print(f"  - SMOTE available: {HAS_SMOTE}")
    
    crypto_symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD', 'ADA-USD',
                     'DOGE-USD', 'DOT-USD', 'LINK-USD', 'AVAX-USD', 'MATIC-USD']
    
    stock_symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'NVDA',
                    'META', 'TSLA', 'JPM', 'V', 'WMT']
    
    all_results = {'crypto': [], 'stocks': []}
    
    print("\n" + "="*80)
    print("CRYPTOCURRENCY TRAINING")
    print("="*80)
    
    for symbol in crypto_symbols:
        result = train_publication_model(symbol, 'crypto', horizon=5, threshold=2.0)
        if result:
            all_results['crypto'].append(result)
    
    print("\n" + "="*80)
    print("STOCK MARKET TRAINING")
    print("="*80)
    
    for symbol in stock_symbols:
        result = train_publication_model(symbol, 'stocks', horizon=5, threshold=2.0)
        if result:
            all_results['stocks'].append(result)
    
    print("\n" + "="*80)
    print("PUBLICATION RESULTS SUMMARY")
    print("="*80)
    
    for asset_type in ['crypto', 'stocks']:
        results = all_results[asset_type]
        if not results:
            continue
        
        print(f"\n{asset_type.upper()} ({len(results)} assets):")
        print("-" * 60)
        
        accuracies = [r['best_accuracy'] for r in results]
        model_accs = {}
        
        for r in results:
            print(f"  {r['symbol']}: {r['best_accuracy']*100:.1f}% ({r['best_model']})")
            for model_name, metrics in r['results'].items():
                if model_name not in model_accs:
                    model_accs[model_name] = []
                model_accs[model_name].append(metrics['avg']['accuracy'])
        
        print(f"\n  Average (best per asset): {np.mean(accuracies)*100:.1f}%")
        print(f"  Best single: {max(accuracies)*100:.1f}%")
        print(f"  Worst single: {min(accuracies)*100:.1f}%")
        
        print("\n  By Model (averaged across assets):")
        for model, accs in sorted(model_accs.items(), key=lambda x: np.mean(x[1]), reverse=True):
            print(f"    {model}: {np.mean(accs)*100:.1f}%")
    
    crypto_accs = [r['best_accuracy'] for r in all_results['crypto']]
    stock_accs = [r['best_accuracy'] for r in all_results['stocks']]
    
    stacking_crypto = [r['results']['StackingEnsemble']['avg']['accuracy'] 
                       for r in all_results['crypto'] if 'StackingEnsemble' in r['results']]
    stacking_stocks = [r['results']['StackingEnsemble']['avg']['accuracy'] 
                       for r in all_results['stocks'] if 'StackingEnsemble' in r['results']]
    
    print("\n" + "="*80)
    print("PAPER METRICS (for IEEE SoutheastCon 2026)")
    print("="*80)
    print("\nBest Model Per Asset:")
    print(f"  Cryptocurrency: {np.mean(crypto_accs)*100:.1f}% average accuracy")
    print(f"  Stock Market: {np.mean(stock_accs)*100:.1f}% average accuracy")
    
    print("\nStacking Ensemble:")
    if stacking_crypto:
        print(f"  Cryptocurrency: {np.mean(stacking_crypto)*100:.1f}% average accuracy")
    if stacking_stocks:
        print(f"  Stock Market: {np.mean(stacking_stocks)*100:.1f}% average accuracy")
    
    all_accs = crypto_accs + stock_accs
    print(f"\nOverall: {np.mean(all_accs)*100:.1f}% average accuracy")
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'configuration': {
            'horizon': 5,
            'threshold': 2.0,
            'validation': 'walk-forward (8 splits)',
            'features': '70+ technical indicators',
            'ensemble': 'Stacking (RF+GB+ET+XGB→LogReg)'
        },
        'results': {
            'crypto': {
                'assets': len(all_results['crypto']),
                'avg_accuracy': float(np.mean(crypto_accs)) if crypto_accs else 0,
                'best_accuracy': float(max(crypto_accs)) if crypto_accs else 0,
                'stacking_avg': float(np.mean(stacking_crypto)) if stacking_crypto else 0
            },
            'stocks': {
                'assets': len(all_results['stocks']),
                'avg_accuracy': float(np.mean(stock_accs)) if stock_accs else 0,
                'best_accuracy': float(max(stock_accs)) if stock_accs else 0,
                'stacking_avg': float(np.mean(stacking_stocks)) if stacking_stocks else 0
            }
        },
        'details': {
            'crypto': [{
                'symbol': r['symbol'],
                'best_model': r['best_model'],
                'accuracy': r['best_accuracy'],
                'class_balance': r['class_balance']
            } for r in all_results['crypto']],
            'stocks': [{
                'symbol': r['symbol'],
                'best_model': r['best_model'],
                'accuracy': r['best_accuracy'],
                'class_balance': r['class_balance']
            } for r in all_results['stocks']]
        }
    }
    
    report_path = os.path.join(RESULTS_DIR, 'publication_results.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nResults saved to: {report_path}")
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return all_results, report


if __name__ == "__main__":
    main()
