# Response to Professor Feedback - January 2026

**Document:** IntelliTradeAI IEEE Paper  
**Author:** Danario Edgar II  
**Date:** January 15, 2026  
**Data Source:** `feedback_analysis_results/feedback_analysis_20260115_050515.json`

---

## Summary of Feedback Items Addressed

| # | Feedback Item | Status |
|---|---------------|--------|
| 1 | More accuracy metrics (precision, recall, F1) | ✓ Completed |
| 2 | Synthetic data analysis - is SMOTE creating diversity? | ✓ Completed |
| 3 | Train with/without synthetic data comparison | ✓ Completed |
| 4 | Test data must be original only | ✓ Verified |
| 5 | Updated ablation study | ✓ Completed |

---

## Feedback Item 1: Comprehensive Accuracy Metrics

**Comment:** "We have 83% on accuracy, we need to use more accuracy metrics like precision, recall and F1."

**Response:** Comprehensive metrics from `comprehensive_metrics` in JSON:

| Asset | Accuracy | Precision | Recall | F1 Score | Specificity |
|-------|----------|-----------|--------|----------|-------------|
| AAPL | 68.09% | 0.00% | 0.00% | 0.00% | 100.00% |
| GOOGL | 47.52% | 47.06% | 10.96% | 17.78% | 86.76% |
| MSFT | 80.85% | 30.77% | 18.18% | 22.86% | 92.44% |
| BTC-USD | 74.29% | 55.56% | 34.48% | 42.55% | 89.47% |
| ETH-USD | 61.90% | 55.00% | 26.19% | 35.48% | 85.71% |

**Key Findings:**
- AAPL shows 0% precision/recall/F1 because the model predicts only the majority class
- BTC-USD has the best F1 (42.55%) and recall (34.48%)
- Accuracy alone is misleading for imbalanced datasets

**Code Reference:** `analysis_script.py`, function `comprehensive_metrics_report()`

---

## Feedback Item 2: Synthetic Data Diversity Analysis

**Comment:** "Is the tool making significant changes to the data or just copying it and pasting it?"

**Response:** SMOTE diversity analysis from `smote_diversity` in JSON:

| Asset | Original | Synthetic | Exact Copies | Near Copies | Diverse (>0.1) | Mean Dist |
|-------|----------|-----------|--------------|-------------|----------------|-----------|
| AAPL | 562 | 204 | 0 (0.0%) | 3 (1.5%) | 190 (93.1%) | 1.130 |
| GOOGL | 562 | 156 | 0 (0.0%) | 1 (0.6%) | 146 (93.6%) | 1.137 |
| MSFT | 562 | 212 | 0 (0.0%) | 2 (0.9%) | 197 (92.9%) | 1.089 |
| BTC-USD | 836 | 236 | 0 (0.0%) | 0 (0.0%) | 221 (93.6%) | 1.022 |
| ETH-USD | 836 | 252 | 0 (0.0%) | 1 (0.4%) | 238 (94.4%) | 0.965 |

**Conclusion:**
- **0% exact copies** across all assets
- **93-94% of synthetic samples** have meaningful distance (>0.1) from originals
- **SMOTE creates TRUE interpolations, NOT copies**

---

## Feedback Item 3: With vs Without Synthetic Data Comparison

**Comment:** "Train the model with and without synthetic data. See what the performance is and justify why using the synthetic data."

### Average SMOTE Benefits (from `smote_benefit` in JSON)

| Asset | Acc Change | Precision Change | Recall Change | F1 Change |
|-------|------------|------------------|---------------|-----------|
| AAPL | +0.95% | -2.78% | +2.96% | +5.14% |
| GOOGL | -1.89% | -25.97% | +5.94% | +6.79% |
| MSFT | +3.07% | -0.92% | -7.58% | -1.39% |
| BTC-USD | -3.81% | -4.94% | +9.77% | +5.65% |
| ETH-USD | -1.59% | -5.24% | +5.56% | +5.03% |

### RandomForest Detailed Comparison

| Asset | Without Acc | With Acc | Without F1 | With F1 |
|-------|-------------|----------|------------|---------|
| AAPL | 68.1% | 68.1% | 0.0% | 0.0% |
| GOOGL | 50.4% | 47.5% | 7.9% | 17.8% |
| MSFT | 81.6% | 80.9% | 18.8% | 22.9% |
| BTC-USD | 73.3% | 74.3% | 24.3% | 42.6% |
| ETH-USD | 61.4% | 61.9% | 18.2% | 35.5% |

### Honest Assessment

| Asset | Accuracy Impact | F1 Impact | SMOTE Beneficial? |
|-------|-----------------|-----------|-------------------|
| AAPL | +0.95% | +5.14% | Marginally |
| GOOGL | -1.89% | +6.79% | Mixed (F1 up, Acc down) |
| MSFT | +3.07% | -1.39% | Mixed (Acc up, F1 down) |
| BTC-USD | -3.81% | +5.65% | Mixed (F1 up, Acc down) |
| ETH-USD | -1.59% | +5.03% | Mixed (F1 up, Acc down) |

**Conclusion:** SMOTE results are **mixed and asset-dependent**. It generally improves F1 (+5-7%) but may reduce accuracy (-2 to -4%). For applications prioritizing signal detection (recall/F1), SMOTE may be beneficial.

---

## Feedback Item 4: Test Data Verification

**Comment:** "Test data needs to be on original data only."

**Response:** **CONFIRMED - Test data is 100% ORIGINAL.**

```python
# Time-series split before SMOTE (Lines 401-402)
X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# SMOTE applied ONLY to training data (Lines 413-416)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
# X_test and y_test are NEVER modified
```

---

## Feedback Item 5: Updated Ablation Study

**Comment:** "For the ablation study, it will need to be updated based on 4."

### AAPL Ablation (141 test samples, original only)

| Configuration | Accuracy | F1 Score | Acc Drop |
|--------------|----------|----------|----------|
| Full Model | 68.09% | 0.00% | - |
| Without Momentum | 68.09% | 0.00% | +0.00% |
| Without Trend | 68.79% | 4.35% | -0.71% |
| Without Volatility | 68.09% | 0.00% | +0.00% |
| Without Volume | 68.09% | 0.00% | +0.00% |

### MSFT Ablation (141 test samples, original only)

| Configuration | Accuracy | F1 Score | Acc Drop |
|--------------|----------|----------|----------|
| Full Model | 81.56% | 18.75% | - |
| Without Momentum | 74.47% | 33.33% | **+7.09%** |
| Without Trend | 84.40% | 0.00% | -2.84% |
| Without Volatility | 82.27% | 0.00% | -0.71% |
| Without Volume | 81.56% | 7.14% | +0.00% |

### BTC-USD Ablation (original test samples only)

| Configuration | Accuracy | F1 Score | Acc Drop |
|--------------|----------|----------|----------|
| Full Model | 73.33% | 24.32% | - |
| Without Momentum | 72.38% | 25.64% | +0.95% |
| Without Trend | 72.38% | 17.14% | +0.95% |
| Without Volatility | 72.38% | 23.68% | +0.95% |
| Without Volume | 75.71% | 33.77% | **-2.38%** |

### ETH-USD Ablation (210 test samples, original only)

| Configuration | Accuracy | F1 Score | Acc Drop |
|--------------|----------|----------|----------|
| Full Model | 61.43% | 18.18% | - |
| Without Momentum | 60.95% | 26.79% | +0.48% |
| Without Trend | 60.00% | 17.65% | **+1.43%** |
| Without Volatility | 62.38% | 18.56% | -0.95% |
| Without Volume | 61.43% | 29.57% | +0.00% |

**Key Findings:**
- **MSFT:** Momentum features most important (+7.09% drop when removed)
- **ETH-USD:** Trend features most important (+1.43% drop when removed)
- **BTC-USD:** Volume features paradoxically hurt performance (removing improves accuracy)
- Results are **asset-dependent** with no universal "most important" feature group

---

## Artifacts

| File | Description |
|------|-------------|
| `analysis_script.py` | Analysis script |
| `feedback_analysis_results/feedback_analysis_20260115_050515.json` | Raw results (source of all tables above) |
| `docs/Professor_Feedback_Response_Jan2026.md` | This document |

---

## Summary

| Item | Finding |
|------|---------|
| 1. Metrics | Precision 0-56%, Recall 0-34%, F1 0-43% - reveals class imbalance issues |
| 2. SMOTE Diversity | ✓ DIVERSE: 0% copies, 93%+ meaningful interpolations |
| 3. With/Without SMOTE | Mixed: F1 improves (+5-7%) but accuracy may decrease (-2 to -4%) |
| 4. Test Data | ✓ VERIFIED: 100% original, never augmented |
| 5. Ablation | Asset-dependent; momentum/trend most important depending on asset |

**All values in this document are extracted directly from the JSON results file.**
