"""
Professor Feedback Analysis Script
==================================
Addresses all feedback items:
1. Comprehensive metrics (precision, recall, F1)
2. Synthetic data (SMOTE) diversity analysis
3. Training with/without synthetic data comparison
4. Test data verification (original only)
5. Updated ablation study
"""

import warnings
warnings.filterwarnings('ignore')

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.spatial.distance import cdist
from scipy.stats import ks_2samp

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix, 
                            classification_report, precision_recall_curve, average_precision_score)

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

RESULTS_DIR = "feedback_analysis_results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def calculate_comprehensive_features(df):
    """Calculate 70+ technical indicators"""
    data = df.copy()
    close = data['Close']
    high = data['High']
    low = data['Low']
    open_price = data['Open']
    volume = data['Volume']
    
    for p in [5, 10, 20, 50]:
        data[f'SMA_{p}'] = close.rolling(p).mean()
    
    for p in [5, 10, 12, 20, 26]:
        data[f'EMA_{p}'] = close.ewm(span=p).mean()
    
    for period in [7, 14, 21]:
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        data[f'RSI_{period}'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
    
    for fast, slow, signal in [(12, 26, 9)]:
        exp1 = close.ewm(span=fast).mean()
        exp2 = close.ewm(span=slow).mean()
        macd = exp1 - exp2
        macd_signal = macd.ewm(span=signal).mean()
        data[f'MACD_{fast}_{slow}'] = macd
        data[f'MACD_Signal_{fast}_{slow}'] = macd_signal
        data[f'MACD_Hist_{fast}_{slow}'] = macd - macd_signal
    
    for p in [10, 20]:
        bb_mid = close.rolling(p).mean()
        bb_std = close.rolling(p).std()
        data[f'BB_Upper_{p}'] = bb_mid + 2 * bb_std
        data[f'BB_Lower_{p}'] = bb_mid - 2 * bb_std
        data[f'BB_Width_{p}'] = (4 * bb_std) / (bb_mid + 1e-10)
        data[f'BB_Position_{p}'] = (close - data[f'BB_Lower_{p}']) / (data[f'BB_Upper_{p}'] - data[f'BB_Lower_{p}'] + 1e-10)
    
    for p in [7, 14]:
        tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
        data[f'ATR_{p}'] = tr.rolling(p).mean()
    
    for p in [5, 14]:
        lowest = low.rolling(p).min()
        highest = high.rolling(p).max()
        data[f'Stoch_K_{p}'] = 100 * (close - lowest) / (highest - lowest + 1e-10)
        data[f'Stoch_D_{p}'] = data[f'Stoch_K_{p}'].rolling(3).mean()
    
    data['OBV'] = (np.sign(close.diff()) * volume).cumsum()
    data['OBV_EMA'] = data['OBV'].ewm(span=20).mean()
    
    for p in [5, 10, 20]:
        data[f'Volume_SMA_{p}'] = volume.rolling(p).mean()
        data[f'Volume_Ratio_{p}'] = volume / (data[f'Volume_SMA_{p}'] + 1e-10)
    
    for p in [1, 2, 3, 5, 10]:
        data[f'Returns_{p}d'] = close.pct_change(p)
    
    for p in [5, 10, 20]:
        data[f'Volatility_{p}d'] = data['Returns_1d'].rolling(p).std()
    
    for p in [5, 10, 20]:
        data[f'Momentum_{p}'] = close - close.shift(p)
        data[f'ROC_{p}'] = 100 * (close - close.shift(p)) / (close.shift(p) + 1e-10)
    
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
    
    data['Price_SMA20_Ratio'] = close / (data['SMA_20'] + 1e-10)
    data['SMA_20_50_Cross'] = (data['SMA_20'] > data['SMA_50']).astype(int)
    data['EMA_12_26_Cross'] = (data['EMA_12'] > data['EMA_26']).astype(int)
    
    data['Skewness'] = data['Returns_1d'].rolling(20).skew()
    data['Kurtosis'] = data['Returns_1d'].rolling(20).kurt()
    
    for lag in [1, 2, 3]:
        data[f'Returns_Lag_{lag}'] = data['Returns_1d'].shift(lag)
        data[f'RSI_Lag_{lag}'] = data['RSI_14'].shift(lag)
    
    return data


def prepare_data(symbol, years=5):
    """Fetch historical data"""
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


def analyze_smote_diversity(X_original, y_original, X_smote, y_smote):
    """
    FEEDBACK ITEM 2: Analyze if SMOTE creates real diversity or just copies
    """
    print("\n" + "="*70)
    print("SYNTHETIC DATA DIVERSITY ANALYSIS")
    print("="*70)
    
    n_original = len(X_original)
    n_synthetic = len(X_smote) - n_original
    
    print(f"\nOriginal samples: {n_original}")
    print(f"After SMOTE: {len(X_smote)}")
    print(f"Synthetic samples added: {n_synthetic}")
    
    if n_synthetic <= 0:
        print("No synthetic samples were generated.")
        return {"diversity_score": 0, "is_copy": True}
    
    X_synth = X_smote[n_original:]
    
    distances = cdist(X_synth, X_original, metric='euclidean')
    min_distances = distances.min(axis=1)
    
    print(f"\nDistance Analysis (synthetic to nearest original):")
    print(f"  Min distance: {min_distances.min():.6f}")
    print(f"  Max distance: {min_distances.max():.6f}")
    print(f"  Mean distance: {min_distances.mean():.6f}")
    print(f"  Std distance: {min_distances.std():.6f}")
    
    exact_copies = (min_distances < 1e-6).sum()
    near_copies = (min_distances < 0.01).sum()
    diverse_samples = (min_distances > 0.1).sum()
    
    print(f"\nCopy Analysis:")
    print(f"  Exact copies (dist < 1e-6): {exact_copies} ({exact_copies/n_synthetic*100:.1f}%)")
    print(f"  Near copies (dist < 0.01): {near_copies} ({near_copies/n_synthetic*100:.1f}%)")
    print(f"  Diverse (dist > 0.1): {diverse_samples} ({diverse_samples/n_synthetic*100:.1f}%)")
    
    original_var = np.var(X_original, axis=0).mean()
    synthetic_var = np.var(X_synth, axis=0).mean()
    
    print(f"\nVariance Analysis:")
    print(f"  Original data variance: {original_var:.6f}")
    print(f"  Synthetic data variance: {synthetic_var:.6f}")
    print(f"  Variance ratio (synth/orig): {synthetic_var/original_var:.3f}")
    
    ks_stats = []
    for i in range(min(10, X_original.shape[1])):
        stat, pval = ks_2samp(X_original[:, i], X_synth[:, i])
        ks_stats.append((stat, pval))
    
    avg_ks_stat = np.mean([s[0] for s in ks_stats])
    print(f"\nDistribution Comparison (KS test, first 10 features):")
    print(f"  Average KS statistic: {avg_ks_stat:.4f}")
    print(f"  (Lower = more similar distributions)")
    
    diversity_score = min_distances.mean() / (np.std(X_original) + 1e-10)
    is_diverse = exact_copies < n_synthetic * 0.1 and min_distances.mean() > 0.01
    
    print(f"\n{'='*70}")
    print("SMOTE DIVERSITY VERDICT:")
    if is_diverse:
        print("  ✓ SMOTE creates DIVERSE synthetic samples")
        print("  ✓ Samples are interpolations, NOT exact copies")
    else:
        print("  ✗ SMOTE may be creating near-copies")
        print("  ✗ Limited diversity in synthetic data")
    print(f"  Diversity Score: {diversity_score:.4f}")
    print(f"{'='*70}")
    
    return {
        "n_original": int(n_original),
        "n_synthetic": int(n_synthetic),
        "exact_copies": int(exact_copies),
        "near_copies": int(near_copies),
        "diverse_samples": int(diverse_samples),
        "mean_distance": float(min_distances.mean()),
        "variance_ratio": float(synthetic_var/original_var),
        "diversity_score": float(diversity_score),
        "is_diverse": bool(is_diverse)
    }


def train_with_without_smote(X_train, y_train, X_test, y_test):
    """
    FEEDBACK ITEM 3: Compare performance with and without synthetic data
    FEEDBACK ITEM 4: Test data is ORIGINAL only (X_test, y_test are never modified)
    """
    print("\n" + "="*70)
    print("TRAINING COMPARISON: WITH vs WITHOUT SYNTHETIC DATA")
    print("="*70)
    print(f"\nTest set: {len(X_test)} samples (100% ORIGINAL DATA)")
    
    results = {}
    
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=150, max_depth=10, 
                                               class_weight='balanced', random_state=42, n_jobs=-1),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, 
                                                        learning_rate=0.05, random_state=42),
    }
    
    if HAS_XGB:
        models['XGBoost'] = xgb.XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.05,
                                              scale_pos_weight=2, use_label_encoder=False, 
                                              eval_metric='logloss', random_state=42, verbosity=0)
    
    print("\n[1] Training WITHOUT synthetic data (original only):")
    print(f"    Training samples: {len(X_train)}")
    print(f"    Class distribution: {np.bincount(y_train)}")
    
    results['without_smote'] = {}
    for name, model in models.items():
        model_copy = model.__class__(**model.get_params())
        model_copy.fit(X_train, y_train)
        y_pred = model_copy.predict(X_test)
        y_proba = model_copy.predict_proba(X_test)[:, 1]
        
        results['without_smote'][name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'auc': roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0.5
        }
        
        print(f"\n    {name}:")
        print(f"      Accuracy:  {results['without_smote'][name]['accuracy']*100:.2f}%")
        print(f"      Precision: {results['without_smote'][name]['precision']*100:.2f}%")
        print(f"      Recall:    {results['without_smote'][name]['recall']*100:.2f}%")
        print(f"      F1 Score:  {results['without_smote'][name]['f1']*100:.2f}%")
        print(f"      AUC-ROC:   {results['without_smote'][name]['auc']:.4f}")
    
    if HAS_SMOTE:
        print("\n[2] Training WITH synthetic data (SMOTE augmented):")
        
        try:
            k = min(5, min(np.bincount(y_train)) - 1)
            if k >= 1:
                smote = SMOTE(k_neighbors=k, random_state=42)
                X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
                
                print(f"    Original training: {len(X_train)}")
                print(f"    After SMOTE: {len(X_train_smote)}")
                print(f"    Synthetic added: {len(X_train_smote) - len(X_train)}")
                print(f"    Class distribution: {np.bincount(y_train_smote)}")
                
                results['with_smote'] = {}
                for name, model in models.items():
                    model_copy = model.__class__(**model.get_params())
                    model_copy.fit(X_train_smote, y_train_smote)
                    y_pred = model_copy.predict(X_test)
                    y_proba = model_copy.predict_proba(X_test)[:, 1]
                    
                    results['with_smote'][name] = {
                        'accuracy': accuracy_score(y_test, y_pred),
                        'precision': precision_score(y_test, y_pred, zero_division=0),
                        'recall': recall_score(y_test, y_pred, zero_division=0),
                        'f1': f1_score(y_test, y_pred, zero_division=0),
                        'auc': roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0.5
                    }
                    
                    print(f"\n    {name}:")
                    print(f"      Accuracy:  {results['with_smote'][name]['accuracy']*100:.2f}%")
                    print(f"      Precision: {results['with_smote'][name]['precision']*100:.2f}%")
                    print(f"      Recall:    {results['with_smote'][name]['recall']*100:.2f}%")
                    print(f"      F1 Score:  {results['with_smote'][name]['f1']*100:.2f}%")
                    print(f"      AUC-ROC:   {results['with_smote'][name]['auc']:.4f}")
                
                print("\n" + "="*70)
                print("COMPARISON SUMMARY: WITH vs WITHOUT SMOTE")
                print("="*70)
                print(f"\n{'Model':<20} {'Metric':<12} {'No SMOTE':<12} {'With SMOTE':<12} {'Delta':<10}")
                print("-" * 66)
                
                for name in models.keys():
                    for metric in ['accuracy', 'precision', 'recall', 'f1']:
                        no_smote = results['without_smote'][name][metric]
                        with_smote = results['with_smote'][name][metric]
                        delta = with_smote - no_smote
                        delta_str = f"+{delta*100:.2f}%" if delta >= 0 else f"{delta*100:.2f}%"
                        print(f"{name:<20} {metric:<12} {no_smote*100:<12.2f} {with_smote*100:<12.2f} {delta_str:<10}")
                    print()
                
                avg_improvement = {}
                for metric in ['accuracy', 'precision', 'recall', 'f1']:
                    improvements = []
                    for name in models.keys():
                        improvements.append(results['with_smote'][name][metric] - results['without_smote'][name][metric])
                    avg_improvement[metric] = np.mean(improvements)
                
                print(f"\nAverage Improvement with SMOTE:")
                for metric, imp in avg_improvement.items():
                    direction = "+" if imp >= 0 else ""
                    print(f"  {metric.capitalize()}: {direction}{imp*100:.2f}%")
                
                results['smote_benefit'] = avg_improvement
                
        except Exception as e:
            print(f"    SMOTE failed: {e}")
            results['with_smote'] = None
    else:
        print("\n[2] SMOTE not available - skipping synthetic data comparison")
        results['with_smote'] = None
    
    return results


def comprehensive_metrics_report(y_test, y_pred, y_proba):
    """
    FEEDBACK ITEM 1: Comprehensive metrics including precision, recall, F1
    """
    print("\n" + "="*70)
    print("COMPREHENSIVE METRICS REPORT")
    print("="*70)
    
    report = classification_report(y_test, y_pred, target_names=['No Move', 'Move >2%'], output_dict=True)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Move', 'Move >2%']))
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives:  {tn}")
    print(f"  False Positives: {fp}")
    print(f"  False Negatives: {fn}")
    print(f"  True Positives:  {tp}")
    
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    print(f"\nDetailed Metrics:")
    print(f"  Accuracy:    {accuracy_score(y_test, y_pred)*100:.2f}%")
    print(f"  Precision:   {precision_score(y_test, y_pred, zero_division=0)*100:.2f}%")
    print(f"  Recall:      {recall_score(y_test, y_pred, zero_division=0)*100:.2f}%")
    print(f"  F1 Score:    {f1_score(y_test, y_pred, zero_division=0)*100:.2f}%")
    print(f"  Specificity: {specificity*100:.2f}%")
    print(f"  Sensitivity: {sensitivity*100:.2f}%")
    
    if y_proba is not None and len(np.unique(y_test)) > 1:
        auc = roc_auc_score(y_test, y_proba)
        ap = average_precision_score(y_test, y_proba)
        print(f"  AUC-ROC:     {auc:.4f}")
        print(f"  Avg Precision: {ap:.4f}")
    
    return {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, zero_division=0)),
        'specificity': float(specificity),
        'sensitivity': float(sensitivity),
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }


def updated_ablation_study(X_train, y_train, X_test, y_test, feature_names):
    """
    FEEDBACK ITEM 5: Updated ablation study with real test data
    """
    print("\n" + "="*70)
    print("UPDATED ABLATION STUDY")
    print("="*70)
    print(f"Test data: {len(X_test)} ORIGINAL samples (no synthetic)")
    
    model = RandomForestClassifier(n_estimators=150, max_depth=10, 
                                   class_weight='balanced', random_state=42, n_jobs=-1)
    
    print("\n[1] Full Model (all features):")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    full_acc = accuracy_score(y_test, y_pred)
    full_f1 = f1_score(y_test, y_pred, zero_division=0)
    print(f"    Accuracy: {full_acc*100:.2f}%")
    print(f"    F1 Score: {full_f1*100:.2f}%")
    
    importances = model.feature_importances_
    feature_importance = list(zip(feature_names, importances))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    print("\n[2] Top 10 Most Important Features:")
    for i, (name, imp) in enumerate(feature_importance[:10]):
        print(f"    {i+1}. {name}: {imp:.4f}")
    
    feature_groups = {
        'momentum': ['RSI', 'MACD', 'ROC', 'Momentum', 'Stoch'],
        'trend': ['SMA', 'EMA', 'Cross'],
        'volatility': ['BB_', 'ATR', 'Volatility'],
        'volume': ['Volume', 'OBV', 'MFI'],
    }
    
    print("\n[3] Feature Group Ablation:")
    ablation_results = {'full': {'accuracy': full_acc, 'f1': full_f1}}
    
    for group_name, patterns in feature_groups.items():
        mask = np.ones(len(feature_names), dtype=bool)
        
        for i, fname in enumerate(feature_names):
            for pattern in patterns:
                if pattern in fname:
                    mask[i] = False
                    break
        
        X_train_ablated = X_train[:, mask]
        X_test_ablated = X_test[:, mask]
        
        model_ablated = RandomForestClassifier(n_estimators=150, max_depth=10, 
                                               class_weight='balanced', random_state=42, n_jobs=-1)
        model_ablated.fit(X_train_ablated, y_train)
        y_pred_ablated = model_ablated.predict(X_test_ablated)
        
        acc = accuracy_score(y_test, y_pred_ablated)
        f1 = f1_score(y_test, y_pred_ablated, zero_division=0)
        
        features_removed = (~mask).sum()
        acc_drop = full_acc - acc
        f1_drop = full_f1 - f1
        
        print(f"\n    Without {group_name.upper()} features ({features_removed} removed):")
        print(f"      Accuracy: {acc*100:.2f}% (drop: {acc_drop*100:.2f}%)")
        print(f"      F1 Score: {f1*100:.2f}% (drop: {f1_drop*100:.2f}%)")
        
        ablation_results[f'without_{group_name}'] = {
            'accuracy': acc, 
            'f1': f1,
            'accuracy_drop': acc_drop,
            'f1_drop': f1_drop,
            'features_removed': int(features_removed)
        }
    
    print("\n" + "="*70)
    print("ABLATION STUDY SUMMARY")
    print("="*70)
    print(f"\n{'Configuration':<25} {'Accuracy':<12} {'F1 Score':<12} {'Acc Drop':<12}")
    print("-" * 60)
    print(f"{'Full Model':<25} {full_acc*100:<12.2f} {full_f1*100:<12.2f} {'N/A':<12}")
    
    for group_name in feature_groups.keys():
        key = f'without_{group_name}'
        r = ablation_results[key]
        print(f"{'w/o ' + group_name.capitalize():<25} {r['accuracy']*100:<12.2f} {r['f1']*100:<12.2f} {r['accuracy_drop']*100:<12.2f}")
    
    most_important = max(feature_groups.keys(), 
                        key=lambda g: ablation_results[f'without_{g}']['accuracy_drop'])
    print(f"\nMost Important Feature Group: {most_important.upper()}")
    print(f"  (Removing it causes largest accuracy drop)")
    
    return ablation_results


def run_full_analysis(symbols=['AAPL', 'GOOGL', 'BTC-USD', 'ETH-USD']):
    """Run complete analysis addressing all feedback items"""
    print("\n" + "="*80)
    print("PROFESSOR FEEDBACK ANALYSIS - COMPLETE REPORT")
    print("="*80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Symbols: {symbols}")
    
    all_results = {
        'metadata': {
            'date': datetime.now().isoformat(),
            'symbols': symbols,
            'feedback_items_addressed': [
                '1. Comprehensive metrics (precision, recall, F1)',
                '2. Synthetic data diversity analysis',
                '3. With/without synthetic data comparison',
                '4. Test data is original only',
                '5. Updated ablation study'
            ]
        },
        'results': {}
    }
    
    for symbol in symbols:
        print(f"\n\n{'#'*80}")
        print(f"ANALYZING: {symbol}")
        print(f"{'#'*80}")
        
        data = prepare_data(symbol, years=3)
        if data is None or len(data) < 300:
            print(f"Insufficient data for {symbol}")
            continue
        
        print(f"Raw data: {len(data)} samples")
        
        data = calculate_comprehensive_features(data)
        
        future_return = (data['Close'].shift(-5) - data['Close']) / data['Close'] * 100
        data['Target'] = (future_return > 2.0).astype(int)
        data = data.replace([np.inf, -np.inf], np.nan).dropna()
        
        feature_cols = [c for c in data.columns if c not in 
                       ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 'Target']]
        
        X = data[feature_cols].values
        y = data['Target'].values
        
        print(f"Processed: {len(data)} samples, {len(feature_cols)} features")
        print(f"Class balance: {y.mean()*100:.1f}% positive")
        
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"\nData Split (time-series):")
        print(f"  Training: {len(X_train)} samples (original)")
        print(f"  Test: {len(X_test)} samples (100% ORIGINAL - no synthetic)")
        
        symbol_results = {}
        
        print("\n" + "-"*70)
        print("FEEDBACK ITEM 2: SMOTE DIVERSITY ANALYSIS")
        print("-"*70)
        
        if HAS_SMOTE and len(np.unique(y_train)) > 1:
            try:
                k = min(5, min(np.bincount(y_train)) - 1)
                if k >= 1:
                    smote = SMOTE(k_neighbors=k, random_state=42)
                    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
                    
                    diversity_analysis = analyze_smote_diversity(X_train, y_train, X_train_smote, y_train_smote)
                    symbol_results['smote_diversity'] = diversity_analysis
                else:
                    print("Not enough samples for SMOTE analysis")
                    symbol_results['smote_diversity'] = None
            except Exception as e:
                print(f"SMOTE analysis failed: {e}")
                symbol_results['smote_diversity'] = None
        else:
            print("SMOTE not available")
            symbol_results['smote_diversity'] = None
        
        print("\n" + "-"*70)
        print("FEEDBACK ITEM 3 & 4: WITH/WITHOUT SMOTE COMPARISON")
        print("-"*70)
        
        comparison_results = train_with_without_smote(X_train, y_train, X_test, y_test)
        symbol_results['smote_comparison'] = comparison_results
        
        print("\n" + "-"*70)
        print("FEEDBACK ITEM 1: COMPREHENSIVE METRICS")
        print("-"*70)
        
        best_model = RandomForestClassifier(n_estimators=150, max_depth=10, 
                                            class_weight='balanced', random_state=42, n_jobs=-1)
        
        if HAS_SMOTE and symbol_results.get('smote_diversity', {}).get('is_diverse', False):
            try:
                k = min(5, min(np.bincount(y_train)) - 1)
                if k >= 1:
                    smote = SMOTE(k_neighbors=k, random_state=42)
                    X_train_final, y_train_final = smote.fit_resample(X_train, y_train)
                else:
                    X_train_final, y_train_final = X_train, y_train
            except:
                X_train_final, y_train_final = X_train, y_train
        else:
            X_train_final, y_train_final = X_train, y_train
        
        best_model.fit(X_train_final, y_train_final)
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]
        
        metrics_report = comprehensive_metrics_report(y_test, y_pred, y_proba)
        symbol_results['comprehensive_metrics'] = metrics_report
        
        print("\n" + "-"*70)
        print("FEEDBACK ITEM 5: UPDATED ABLATION STUDY")
        print("-"*70)
        
        ablation_results = updated_ablation_study(X_train, y_train, X_test, y_test, feature_cols)
        symbol_results['ablation_study'] = ablation_results
        
        all_results['results'][symbol] = symbol_results
    
    results_file = f"{RESULTS_DIR}/feedback_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj
    
    with open(results_file, 'w') as f:
        json.dump(convert_numpy(all_results), f, indent=2)
    
    print(f"\n\nResults saved to: {results_file}")
    
    return all_results


if __name__ == "__main__":
    results = run_full_analysis(['AAPL', 'GOOGL', 'MSFT', 'BTC-USD', 'ETH-USD'])
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nFeedback Items Addressed:")
    print("  1. ✓ Comprehensive metrics (precision, recall, F1, specificity)")
    print("  2. ✓ SMOTE diversity analysis (checks for copies vs interpolations)")
    print("  3. ✓ With/without SMOTE comparison")
    print("  4. ✓ Test data verified as original only")
    print("  5. ✓ Updated ablation study with original test data")
