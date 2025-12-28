# Response to Reviewer Comment: "Overall result is not promising..."

## Reviewer Comment (MS29)
"Overall result is not promising..."

---

## Author Response

We thank the reviewer for this constructive feedback. We have significantly improved our model architecture and experimental results to address this concern.

### Summary of Improvements

In the revised manuscript, we enhanced our machine learning pipeline with the following modifications:

1. **Stacking Ensemble Architecture**: We upgraded from a simple voting ensemble to a stacking ensemble combining four complementary base learners:
   - Enhanced Bidirectional LSTM with attention mechanism (3 layers: 128-64-32 units)
   - Random Forest (200 trees, depth=15)
   - XGBoost (300 boosting rounds)
   - LightGBM (300 rounds, 31 leaves)
   
   A logistic regression meta-learner combines out-of-fold predictions from all base models.

2. **Class Balancing**: We applied Synthetic Minority Over-sampling Technique (SMOTE) to address class imbalance between up/down price movements, using k=5 nearest neighbors for synthetic sample generation.

3. **Temporal Data Handling**: We replaced standard k-fold cross-validation with TimeSeriesSplit to prevent data leakage, ensuring training data always precedes validation data temporally.

4. **Hyperparameter Optimization**: We implemented Bayesian hyperparameter optimization using the Optuna framework with Tree-structured Parzen Estimator sampling (50 trials per model).

### Improved Results

| Metric | Previous | Revised | Improvement |
|--------|----------|---------|-------------|
| Cryptocurrency Accuracy | 68.2% | 73.4% | +5.2 pp |
| Stock Accuracy | 71.5% | 75.8% | +4.3 pp |
| AUC-ROC | 0.741 | 0.812 | +0.071 |
| Sharpe Ratio | 1.74 | 1.92 | +10.3% |

The tri-signal fusion approach now achieves an **8.2 percentage point improvement (12.6% relative)** over baseline ML-only predictions, compared to the previous 5.4 pp improvement.

### Context and Justification

We acknowledge that prediction accuracies of 73-76% may appear modest compared to classification tasks in other domains. However, financial market prediction is inherently challenging due to:

1. **Market Efficiency**: According to the Efficient Market Hypothesis, asset prices already reflect available information, making prediction difficult.

2. **Literature Benchmarks**: Our results exceed competitive benchmarks in the financial prediction literature, where accuracies of 55-70% are considered competitive (Jiang, 2021).

3. **Practical Significance**: Even modest improvements in prediction accuracy translate to meaningful financial returns. Our backtesting demonstrates that the improved model achieves a Sharpe ratio of 1.92 with maximum drawdown of only -10.5%, outperforming both the S&P 500 benchmark and baseline approaches.

### Revised Sections

The following sections have been updated to reflect these improvements:
- Abstract (lines 29-31)
- Section III.B: Machine Learning Models
- Section III.C: Class Balancing and Cross-Validation (new subsection)
- Section IV: Results (Tables IV and V)
- Section VI: Conclusion

We believe these substantial improvements address the reviewer's concern about the overall results, demonstrating that our tri-signal fusion approach with stacking ensemble achieves competitive performance for the challenging task of financial market prediction.
