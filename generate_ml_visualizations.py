"""
ML Training Visualization Generator
Creates diagrams and charts for ML documentation
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from datetime import datetime, timedelta
import pandas as pd

# Set style
plt.style.use('seaborn-v0_8-darkgrid')

def create_confusion_matrix_visualization():
    """Create confusion matrix heatmap"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Sample confusion matrix data
    confusion_matrix = np.array([[45, 12], [8, 55]])
    
    # Create heatmap
    im = ax.imshow(confusion_matrix, cmap='Blues', alpha=0.8)
    
    # Labels
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicted DOWN (0)', 'Predicted UP (1)'], fontsize=12)
    ax.set_yticklabels(['Actual DOWN (0)', 'Actual UP (1)'], fontsize=12)
    
    # Add values
    for i in range(2):
        for j in range(2):
            value = confusion_matrix[i, j]
            color = 'white' if value > 40 else 'black'
            ax.text(j, i, str(value), ha='center', va='center', 
                   fontsize=24, color=color, weight='bold')
    
    # Labels for each cell
    labels = [
        ['True Negative\n(Correctly predicted DOWN)', 'False Positive\n(Wrongly predicted UP)'],
        ['False Negative\n(Wrongly predicted DOWN)', 'True Positive\n(Correctly predicted UP)']
    ]
    
    for i in range(2):
        for j in range(2):
            ax.text(j, i + 0.3, labels[i][j], ha='center', va='center', 
                   fontsize=9, style='italic', color='gray')
    
    # Title and colorbar
    ax.set_title('Confusion Matrix - Binary Classification\n(Sample: 120 test predictions)', 
                fontsize=16, weight='bold', pad=20)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Count', rotation=270, labelpad=20, fontsize=12)
    
    # Add metrics annotations
    accuracy = (45 + 55) / 120
    precision = 55 / (55 + 8)
    recall = 55 / (55 + 12)
    f1 = 2 * (precision * recall) / (precision + recall)
    
    metrics_text = f"""
    Accuracy:  {accuracy:.1%}
    Precision: {precision:.1%}
    Recall:    {recall:.1%}
    F1 Score:  {f1:.1%}
    """
    
    ax.text(1.45, 0.5, metrics_text, transform=ax.transAxes, 
           fontsize=11, verticalalignment='center',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('diagrams/confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("✅ Created: diagrams/confusion_matrix.png")
    plt.close()


def create_class_distribution_visualization():
    """Visualize target variable distribution"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Sample data (BTC 365 days)
    class_0 = 157  # DOWN days
    class_1 = 208  # UP days
    total = class_0 + class_1
    
    # Bar chart
    classes = ['DOWN (0)', 'UP (1)']
    counts = [class_0, class_1]
    colors = ['#e74c3c', '#2ecc71']
    
    bars = ax1.bar(classes, counts, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax1.set_ylabel('Number of Days', fontsize=14, weight='bold')
    ax1.set_xlabel('Target Class', fontsize=14, weight='bold')
    ax1.set_title('Class Distribution - BTC (365 Days)', fontsize=16, weight='bold')
    ax1.set_ylim(0, 250)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        percentage = (count / total) * 100
        ax1.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{count}\n({percentage:.1f}%)',
                ha='center', va='bottom', fontsize=13, weight='bold')
    
    # Add balance line
    ax1.axhline(y=total/2, color='blue', linestyle='--', linewidth=2, alpha=0.5, 
               label='Perfect Balance (50%)')
    ax1.legend(fontsize=11)
    
    # Pie chart
    explode = (0.05, 0.05)
    wedges, texts, autotexts = ax2.pie(counts, labels=classes, autopct='%1.1f%%',
                                       colors=colors, explode=explode,
                                       shadow=True, startangle=90,
                                       textprops={'fontsize': 12, 'weight': 'bold'})
    
    ax2.set_title('Class Distribution Percentage', fontsize=16, weight='bold')
    
    # Add legend with counts
    legend_labels = [f'{cls}: {cnt} days' for cls, cnt in zip(classes, counts)]
    ax2.legend(legend_labels, loc='lower left', fontsize=11)
    
    # Add balance assessment
    balance_ratio = max(counts) / min(counts)
    if balance_ratio < 1.2:
        balance_status = "✅ Well Balanced"
        status_color = 'green'
    elif balance_ratio < 1.5:
        balance_status = "⚠️ Slightly Imbalanced"
        status_color = 'orange'
    else:
        balance_status = "❌ Imbalanced"
        status_color = 'red'
    
    fig.text(0.5, 0.02, f'Balance Ratio: {balance_ratio:.2f} - {balance_status}',
            ha='center', fontsize=13, weight='bold', color=status_color,
            bbox=dict(boxstyle='round', facecolor='white', edgecolor=status_color, linewidth=2))
    
    plt.tight_layout()
    plt.savefig('diagrams/class_distribution.png', dpi=300, bbox_inches='tight')
    print("✅ Created: diagrams/class_distribution.png")
    plt.close()


def create_train_test_split_visualization():
    """Visualize time-series train/test split"""
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Timeline
    total_days = 365
    train_days = int(total_days * 0.8)  # 292 days
    test_days = total_days - train_days  # 73 days
    
    # Draw rectangles
    train_rect = patches.Rectangle((0, 0.3), train_days, 0.4, 
                                   linewidth=2, edgecolor='blue', 
                                   facecolor='lightblue', alpha=0.7)
    test_rect = patches.Rectangle((train_days, 0.3), test_days, 0.4, 
                                  linewidth=2, edgecolor='red', 
                                  facecolor='lightcoral', alpha=0.7)
    
    ax.add_patch(train_rect)
    ax.add_patch(test_rect)
    
    # Labels
    ax.text(train_days/2, 0.5, f'TRAINING SET\n{train_days} days (80%)\nJan 1 - Oct 19', 
           ha='center', va='center', fontsize=14, weight='bold', color='darkblue')
    ax.text(train_days + test_days/2, 0.5, f'TEST SET\n{test_days} days (20%)\nOct 20 - Dec 31', 
           ha='center', va='center', fontsize=14, weight='bold', color='darkred')
    
    # Timeline axis
    ax.set_xlim(0, total_days)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Days (Chronological Order)', fontsize=14, weight='bold')
    ax.set_title('Time-Series Train/Test Split (80/20 Split)', 
                fontsize=16, weight='bold', pad=20)
    
    # Month markers
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    days_per_month = total_days / 12
    for i, month in enumerate(months):
        x_pos = i * days_per_month
        ax.axvline(x=x_pos, color='gray', linestyle=':', alpha=0.5)
        ax.text(x_pos + days_per_month/2, 0.05, month, ha='center', fontsize=10)
    
    # Split line
    ax.axvline(x=train_days, color='black', linestyle='--', linewidth=3, alpha=0.8)
    ax.text(train_days, 0.85, '← PAST DATA (Learn) | FUTURE DATA (Validate) →', 
           ha='center', fontsize=12, weight='bold',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('diagrams/train_test_split.png', dpi=300, bbox_inches='tight')
    print("✅ Created: diagrams/train_test_split.png")
    plt.close()


def create_metrics_comparison_chart():
    """Compare different ML models"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Sample model performance data
    models = ['Random\nForest', 'XGBoost', 'LSTM', 'Ensemble\n(Combined)']
    accuracy = [78, 83, 76, 85]
    precision = [80, 87, 74, 88]
    recall = [75, 80, 78, 82]
    f1_score = [77, 83, 76, 85]
    
    # Bar width and positions
    x = np.arange(len(models))
    width = 0.2
    
    # Create bars
    bars1 = ax.bar(x - 1.5*width, accuracy, width, label='Accuracy', 
                   color='#3498db', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x - 0.5*width, precision, width, label='Precision', 
                   color='#2ecc71', alpha=0.8, edgecolor='black')
    bars3 = ax.bar(x + 0.5*width, recall, width, label='Recall', 
                   color='#e74c3c', alpha=0.8, edgecolor='black')
    bars4 = ax.bar(x + 1.5*width, f1_score, width, label='F1 Score', 
                   color='#f39c12', alpha=0.8, edgecolor='black')
    
    # Add value labels
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{height}%', ha='center', va='bottom', fontsize=9, weight='bold')
    
    # Styling
    ax.set_ylabel('Score (%)', fontsize=14, weight='bold')
    ax.set_xlabel('Model Type', fontsize=14, weight='bold')
    ax.set_title('Model Performance Comparison\n(All Metrics)', 
                fontsize=16, weight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12, weight='bold')
    ax.set_ylim(0, 100)
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    
    # Add target line
    ax.axhline(y=80, color='green', linestyle='--', linewidth=2, alpha=0.5, 
              label='Target (80%)')
    
    # Highlight best model
    best_idx = 3  # Ensemble
    highlight = patches.Rectangle((x[best_idx] - 2*width, 0), 4*width, 100, 
                                 linewidth=3, edgecolor='gold', facecolor='none', 
                                 linestyle='--')
    ax.add_patch(highlight)
    ax.text(x[best_idx], 95, '★ BEST ★', ha='center', fontsize=12, 
           weight='bold', color='gold')
    
    plt.tight_layout()
    plt.savefig('diagrams/model_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Created: diagrams/model_comparison.png")
    plt.close()


def create_roc_curve_visualization():
    """Create ROC curve"""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Sample ROC curve data (simulated)
    fpr = np.linspace(0, 1, 100)
    
    # Different models
    tpr_rf = 1 - (1 - fpr) ** 1.5  # Random Forest (AUC ~0.82)
    tpr_xgb = 1 - (1 - fpr) ** 1.8  # XGBoost (AUC ~0.89)
    tpr_lstm = 1 - (1 - fpr) ** 1.3  # LSTM (AUC ~0.78)
    tpr_ensemble = 1 - (1 - fpr) ** 2.0  # Ensemble (AUC ~0.91)
    
    # Plot curves
    ax.plot(fpr, tpr_rf, linewidth=3, label='Random Forest (AUC = 0.82)', color='blue')
    ax.plot(fpr, tpr_xgb, linewidth=3, label='XGBoost (AUC = 0.89)', color='green')
    ax.plot(fpr, tpr_lstm, linewidth=3, label='LSTM (AUC = 0.78)', color='red')
    ax.plot(fpr, tpr_ensemble, linewidth=4, label='Ensemble (AUC = 0.91) ★', 
           color='gold', linestyle='--')
    
    # Random classifier line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.5, 
           label='Random Classifier (AUC = 0.50)')
    
    # Styling
    ax.set_xlabel('False Positive Rate (FPR)', fontsize=14, weight='bold')
    ax.set_ylabel('True Positive Rate (TPR)', fontsize=14, weight='bold')
    ax.set_title('ROC Curves - Model Comparison\n(Higher curve = Better model)', 
                fontsize=16, weight='bold', pad=20)
    ax.legend(fontsize=12, loc='lower right')
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    # Add annotations
    ax.text(0.6, 0.2, 'Perfect Classifier\n(AUC = 1.0)', 
           fontsize=11, style='italic', color='gray')
    ax.text(0.5, 0.45, 'Random Guessing\n(AUC = 0.5)', 
           fontsize=11, style='italic', color='gray')
    
    # Add performance zones
    ax.fill_between([0, 1], [0, 1], [1, 1], alpha=0.1, color='green', 
                    label='Good Performance Zone')
    ax.fill_between([0, 1], [0, 0], [0, 1], alpha=0.1, color='red')
    
    plt.tight_layout()
    plt.savefig('diagrams/roc_curve.png', dpi=300, bbox_inches='tight')
    print("✅ Created: diagrams/roc_curve.png")
    plt.close()


def create_feature_importance_chart():
    """Visualize top features"""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Top 15 features (sample data)
    features = [
        'RSI', 'MACD_histogram', 'Volume_ratio', 'ROC_5', 
        'Price_vs_SMA_20', 'BB_position', 'ATR', 'Volume_price_trend',
        'Return_lag_1', 'Stoch_K', 'Williams_R', 'Momentum_10',
        'Volatility_20', 'EMA_12', 'Price_position'
    ]
    importance = [0.142, 0.118, 0.095, 0.087, 0.079, 0.071, 0.063, 0.055,
                 0.048, 0.042, 0.037, 0.033, 0.029, 0.025, 0.021]
    
    # Sort by importance
    sorted_idx = np.argsort(importance)
    features_sorted = [features[i] for i in sorted_idx]
    importance_sorted = [importance[i] for i in sorted_idx]
    
    # Color gradient
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(features)))
    
    # Horizontal bar chart
    bars = ax.barh(range(len(features_sorted)), importance_sorted, 
                   color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for i, (bar, imp) in enumerate(zip(bars, importance_sorted)):
        width = bar.get_width()
        ax.text(width + 0.005, bar.get_y() + bar.get_height()/2,
               f'{imp:.1%}', ha='left', va='center', fontsize=10, weight='bold')
    
    # Styling
    ax.set_yticks(range(len(features_sorted)))
    ax.set_yticklabels(features_sorted, fontsize=11, weight='bold')
    ax.set_xlabel('Feature Importance Score', fontsize=14, weight='bold')
    ax.set_title('Top 15 Most Important Features\n(Random Forest Model)', 
                fontsize=16, weight='bold', pad=20)
    ax.set_xlim(0, max(importance) * 1.15)
    ax.grid(axis='x', alpha=0.3)
    
    # Highlight top 3
    for i in range(len(features_sorted) - 3, len(features_sorted)):
        bars[i].set_edgecolor('gold')
        bars[i].set_linewidth(3)
    
    # Add medal symbols
    medals = ['🥇', '🥈', '🥉']
    for i, medal in enumerate(medals):
        ax.text(-0.01, len(features_sorted) - i - 1, medal, 
               fontsize=16, ha='right', va='center')
    
    plt.tight_layout()
    plt.savefig('diagrams/feature_importance.png', dpi=300, bbox_inches='tight')
    print("✅ Created: diagrams/feature_importance.png")
    plt.close()


def create_rolling_winrate_visualization():
    """Visualize rolling win rate over time"""
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Simulate 365 days of price movements
    np.random.seed(42)
    
    # Create realistic price series (upward bias)
    returns = np.random.normal(0.001, 0.02, 365)  # Slight positive bias
    price = 100 * np.cumprod(1 + returns)
    
    # Binary outcomes (1 = up, 0 = down)
    daily_outcome = (np.diff(price) > 0).astype(int)
    daily_outcome = np.insert(daily_outcome, 0, 1)  # Add first day
    
    # Calculate rolling 30-day win rate
    window = 30
    rolling_winrate = pd.Series(daily_outcome).rolling(window=window).mean() * 100
    
    # Dates
    dates = pd.date_range(start='2024-01-01', periods=365, freq='D')
    
    # Plot
    ax.plot(dates, rolling_winrate, linewidth=3, color='blue', label='30-Day Win Rate')
    ax.fill_between(dates, rolling_winrate, 50, where=(rolling_winrate >= 50), 
                    alpha=0.3, color='green', label='Above 50% (Bullish)')
    ax.fill_between(dates, rolling_winrate, 50, where=(rolling_winrate < 50), 
                    alpha=0.3, color='red', label='Below 50% (Bearish)')
    
    # Reference lines
    ax.axhline(y=50, color='black', linestyle='--', linewidth=2, alpha=0.5, 
              label='50% (Neutral)')
    ax.axhline(y=60, color='green', linestyle=':', linewidth=2, alpha=0.5)
    ax.axhline(y=40, color='red', linestyle=':', linewidth=2, alpha=0.5)
    
    # Styling
    ax.set_xlabel('Date', fontsize=14, weight='bold')
    ax.set_ylabel('Win Rate (%)', fontsize=14, weight='bold')
    ax.set_title('Rolling 30-Day Win Rate - BTC Price Movements\n(% of days price increased)', 
                fontsize=16, weight='bold', pad=20)
    ax.set_ylim(20, 80)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=12, loc='upper left')
    
    # Add statistics
    mean_winrate = rolling_winrate.mean()
    ax.text(0.98, 0.95, f'Mean Win Rate: {mean_winrate:.1f}%', 
           transform=ax.transAxes, fontsize=13, weight='bold',
           ha='right', va='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # Format x-axis
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig('diagrams/rolling_winrate.png', dpi=300, bbox_inches='tight')
    print("✅ Created: diagrams/rolling_winrate.png")
    plt.close()


def create_learning_curve():
    """Create learning curve showing model performance vs training size"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Simulate learning curve data
    train_sizes = np.array([50, 100, 150, 200, 250, 292])
    train_scores = np.array([0.95, 0.96, 0.97, 0.975, 0.98, 0.985])
    test_scores = np.array([0.65, 0.72, 0.78, 0.82, 0.84, 0.85])
    
    # Plot
    ax.plot(train_sizes, train_scores, 'o-', linewidth=3, markersize=10, 
           color='blue', label='Training Accuracy')
    ax.plot(train_sizes, test_scores, 'o-', linewidth=3, markersize=10, 
           color='red', label='Validation Accuracy')
    
    # Fill between
    ax.fill_between(train_sizes, train_scores, test_scores, alpha=0.2, color='yellow')
    
    # Overfitting gap annotation
    gap_idx = 3
    gap = train_scores[gap_idx] - test_scores[gap_idx]
    ax.annotate(f'Overfitting Gap\n{gap:.1%}',
               xy=(train_sizes[gap_idx], (train_scores[gap_idx] + test_scores[gap_idx])/2),
               xytext=(train_sizes[gap_idx] + 50, 0.75),
               arrowprops=dict(arrowstyle='->', color='orange', lw=2),
               fontsize=11, weight='bold', color='orange',
               bbox=dict(boxstyle='round', facecolor='white', edgecolor='orange'))
    
    # Styling
    ax.set_xlabel('Training Set Size (samples)', fontsize=14, weight='bold')
    ax.set_ylabel('Accuracy Score', fontsize=14, weight='bold')
    ax.set_title('Learning Curve - Model Performance vs Training Data Size', 
                fontsize=16, weight='bold', pad=20)
    ax.set_ylim(0.6, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=13, loc='lower right')
    
    # Add target line
    ax.axhline(y=0.80, color='green', linestyle='--', linewidth=2, alpha=0.5,
              label='Target Accuracy (80%)')
    
    # Add interpretation text
    interpretation = """
    Interpretation:
    • Training accuracy increasing → Model learning
    • Test accuracy plateauing → Need more diverse data
    • Gap between curves → Some overfitting present
    • Test score 85% → Good generalization
    """
    ax.text(0.02, 0.98, interpretation, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('diagrams/learning_curve.png', dpi=300, bbox_inches='tight')
    print("✅ Created: diagrams/learning_curve.png")
    plt.close()


if __name__ == '__main__':
    print("🎨 Generating ML Training Visualizations...")
    print("=" * 60)
    
    # Create all visualizations
    create_confusion_matrix_visualization()
    create_class_distribution_visualization()
    create_train_test_split_visualization()
    create_metrics_comparison_chart()
    create_roc_curve_visualization()
    create_feature_importance_chart()
    create_rolling_winrate_visualization()
    create_learning_curve()
    
    print("=" * 60)
    print("✅ All visualizations created successfully!")
    print("\nGenerated files:")
    print("  • diagrams/confusion_matrix.png")
    print("  • diagrams/class_distribution.png")
    print("  • diagrams/train_test_split.png")
    print("  • diagrams/model_comparison.png")
    print("  • diagrams/roc_curve.png")
    print("  • diagrams/feature_importance.png")
    print("  • diagrams/rolling_winrate.png")
    print("  • diagrams/learning_curve.png")
