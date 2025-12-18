import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def create_class_box(ax, x, y, width, height, class_name, attributes, methods, color='#E3F2FD'):
    """Create a UML class box"""
    box = FancyBboxPatch((x, y), width, height,
                         boxstyle="round,pad=0.02",
                         edgecolor='#1565C0', facecolor=color,
                         linewidth=2)
    ax.add_patch(box)
    
    header_height = height * 0.25
    header_box = FancyBboxPatch((x, y + height - header_height), width, header_height,
                                 boxstyle="round,pad=0.02",
                                 edgecolor='#1565C0', facecolor='#1565C0',
                                 linewidth=2)
    ax.add_patch(header_box)
    
    ax.text(x + width/2, y + height - header_height/2, class_name,
            ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    
    attr_y = y + height - header_height - 0.15
    for attr in attributes[:3]:
        ax.text(x + 0.1, attr_y, f"- {attr}", fontsize=6, va='top')
        attr_y -= 0.18
    
    ax.plot([x, x + width], [attr_y + 0.05, attr_y + 0.05], 'k-', linewidth=0.5)
    
    method_y = attr_y - 0.05
    for method in methods[:3]:
        ax.text(x + 0.1, method_y, f"+ {method}()", fontsize=6, va='top')
        method_y -= 0.18

def draw_arrow(ax, start, end, style='->'):
    """Draw connection arrow between classes"""
    arrow = FancyArrowPatch(start, end,
                           arrowstyle=style, mutation_scale=15,
                           linestyle='-', linewidth=1.5, color='#424242')
    ax.add_patch(arrow)

def create_class_diagram():
    """Create comprehensive UML class diagram"""
    fig, ax = plt.subplots(figsize=(20, 14))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 14)
    ax.axis('off')
    ax.set_title('IntelliTradeAI - System Class Diagram', fontsize=16, fontweight='bold', pad=20)
    
    layer_colors = {
        'data': '#E3F2FD',
        'ml': '#F3E5F5',
        'ai': '#E8F5E9',
        'trading': '#FFF3E0',
        'ui': '#FCE4EC',
        'security': '#EFEBE9'
    }
    
    ax.add_patch(FancyBboxPatch((0.3, 10.5), 4.5, 3.2, boxstyle="round,pad=0.1",
                                 edgecolor='#1565C0', facecolor='#E3F2FD', alpha=0.3, linewidth=2))
    ax.text(0.5, 13.5, 'DATA LAYER', fontsize=10, fontweight='bold', color='#1565C0')
    
    ax.add_patch(FancyBboxPatch((5.2, 10.5), 5.5, 3.2, boxstyle="round,pad=0.1",
                                 edgecolor='#7B1FA2', facecolor='#F3E5F5', alpha=0.3, linewidth=2))
    ax.text(5.4, 13.5, 'ML MODELS LAYER', fontsize=10, fontweight='bold', color='#7B1FA2')
    
    ax.add_patch(FancyBboxPatch((11.0, 10.5), 4.5, 3.2, boxstyle="round,pad=0.1",
                                 edgecolor='#388E3C', facecolor='#E8F5E9', alpha=0.3, linewidth=2))
    ax.text(11.2, 13.5, 'AI ADVISOR LAYER', fontsize=10, fontweight='bold', color='#388E3C')
    
    ax.add_patch(FancyBboxPatch((15.8, 10.5), 3.9, 3.2, boxstyle="round,pad=0.1",
                                 edgecolor='#F57C00', facecolor='#FFF3E0', alpha=0.3, linewidth=2))
    ax.text(16.0, 13.5, 'TRADING LAYER', fontsize=10, fontweight='bold', color='#F57C00')
    
    ax.add_patch(FancyBboxPatch((0.3, 5.5), 6, 4.5, boxstyle="round,pad=0.1",
                                 edgecolor='#C2185B', facecolor='#FCE4EC', alpha=0.3, linewidth=2))
    ax.text(0.5, 9.8, 'UI LAYER', fontsize=10, fontweight='bold', color='#C2185B')
    
    ax.add_patch(FancyBboxPatch((6.8, 5.5), 4.5, 4.5, boxstyle="round,pad=0.1",
                                 edgecolor='#5D4037', facecolor='#EFEBE9', alpha=0.3, linewidth=2))
    ax.text(7.0, 9.8, 'SECURITY LAYER', fontsize=10, fontweight='bold', color='#5D4037')
    
    ax.add_patch(FancyBboxPatch((11.8, 5.5), 7.9, 4.5, boxstyle="round,pad=0.1",
                                 edgecolor='#00695C', facecolor='#E0F2F1', alpha=0.3, linewidth=2))
    ax.text(12.0, 9.8, 'DATABASE & COMPLIANCE LAYER', fontsize=10, fontweight='bold', color='#00695C')
    
    create_class_box(ax, 0.5, 11, 4, 2.3, 'DataIngestion',
                    ['coinmarketcap_api_key', 'yahoo_timeout'],
                    ['fetch_crypto_data', 'fetch_stock_data', 'fetch_mixed_data'],
                    layer_colors['data'])
    
    create_class_box(ax, 5.5, 11.5, 2.3, 1.8, 'MLPredictor',
                    ['models', 'scalers'],
                    ['predict', 'get_signal', 'analyze_asset'],
                    layer_colors['ml'])
    
    create_class_box(ax, 8.0, 11.5, 2.3, 1.8, 'XGBoostModel',
                    ['n_estimators', 'learning_rate'],
                    ['train', 'predict', 'predict_proba'],
                    layer_colors['ml'])
    
    create_class_box(ax, 11.3, 11.5, 2, 1.8, 'SignalFusion',
                    ['ml_weight', 'pattern_weight'],
                    ['fuse_signals', 'resolve_conflict'],
                    layer_colors['ai'])
    
    create_class_box(ax, 13.5, 11.5, 2, 1.8, 'PatternRecognizer',
                    ['pattern_library'],
                    ['detect_patterns', 'identify_levels'],
                    layer_colors['ai'])
    
    create_class_box(ax, 16, 12, 3.5, 1.5, 'TradingModeManager',
                    ['current_mode', 'auto_config'],
                    ['switch_mode', 'execute_trade'],
                    layer_colors['trading'])
    
    create_class_box(ax, 16, 10.7, 3.5, 1.2, 'TradeExecutor',
                    ['db_manager'],
                    ['execute_stock', 'execute_crypto'],
                    layer_colors['trading'])
    
    create_class_box(ax, 0.5, 6, 5.5, 3.5, 'EnhancedDashboard',
                    ['session_state', 'ai_advisor', 'signal_fusion'],
                    ['render_main', 'render_trading', 'render_options'],
                    layer_colors['ui'])
    
    create_class_box(ax, 7, 7.5, 4, 2, 'SecureAuthManager',
                    ['secret_key', 'jwt_config'],
                    ['authenticate', 'register', 'verify_2fa'],
                    layer_colors['security'])
    
    create_class_box(ax, 7, 5.8, 4, 1.5, 'SecureWalletManager',
                    ['web3_provider'],
                    ['create_wallet', 'sign_transaction'],
                    layer_colors['security'])
    
    create_class_box(ax, 12, 7.5, 3.5, 2, 'DBManager',
                    ['connection', 'pool'],
                    ['save_trade', 'get_portfolio', 'save_alert'],
                    '#E0F2F1')
    
    create_class_box(ax, 15.8, 7.5, 3.7, 2, 'LegalCompliance',
                    ['risk_disclosures'],
                    ['get_disclosures', 'save_esignature'],
                    '#E0F2F1')
    
    create_class_box(ax, 12, 5.8, 3.5, 1.5, 'UserOnboarding',
                    ['risk_levels'],
                    ['create_profile', 'get_trading_plan'],
                    '#E0F2F1')
    
    create_class_box(ax, 15.8, 5.8, 3.7, 1.5, 'OptionsDataFetcher',
                    ['yahoo_client'],
                    ['get_chain', 'get_recommendations'],
                    '#E0F2F1')
    
    draw_arrow(ax, (4.5, 12), (5.5, 12))
    draw_arrow(ax, (7.8, 12.3), (8, 12.3))
    draw_arrow(ax, (10.3, 12.3), (11.3, 12.3))
    draw_arrow(ax, (13.3, 12.3), (13.5, 12.3))
    draw_arrow(ax, (15.5, 12.3), (16, 12.3))
    
    draw_arrow(ax, (3, 9.5), (3, 11))
    draw_arrow(ax, (6, 9), (6.5, 11.5))
    draw_arrow(ax, (6, 8.5), (12.3, 11.5))
    
    draw_arrow(ax, (11, 8.5), (12, 8.5))
    draw_arrow(ax, (11, 7), (12, 7))
    draw_arrow(ax, (15.5, 8.5), (15.8, 8.5))
    
    ax.add_patch(FancyBboxPatch((0.3, 0.5), 19.4, 4.5, boxstyle="round,pad=0.1",
                                 edgecolor='#455A64', facecolor='#ECEFF1', alpha=0.3, linewidth=2))
    ax.text(0.5, 4.8, 'TRADING PLAN & FEATURES', fontsize=10, fontweight='bold', color='#455A64')
    
    create_class_box(ax, 0.5, 1, 3.5, 2, 'TradingPlan',
                    ['risk_tier', 'investment_amount'],
                    ['generate_plan', 'get_allocations'],
                    '#ECEFF1')
    
    create_class_box(ax, 4.3, 1, 3.5, 2, 'SectorRankings',
                    ['sectors', 'etfs'],
                    ['get_rankings', 'render_table'],
                    '#ECEFF1')
    
    create_class_box(ax, 8.1, 1, 3.5, 2, 'PriceAlerts',
                    ['alerts', 'thresholds'],
                    ['add_alert', 'check_alerts'],
                    '#ECEFF1')
    
    create_class_box(ax, 11.9, 1, 3.5, 2, 'OptionsRecommendations',
                    ['tier_strategies'],
                    ['get_suggestions', 'render_options'],
                    '#ECEFF1')
    
    create_class_box(ax, 15.7, 1, 3.5, 2, 'TooltipDefinitions',
                    ['terms', 'categories'],
                    ['get_definition', 'inject_tooltips'],
                    '#ECEFF1')
    
    legend_elements = [
        mpatches.Patch(facecolor='#E3F2FD', edgecolor='#1565C0', label='Data Layer'),
        mpatches.Patch(facecolor='#F3E5F5', edgecolor='#7B1FA2', label='ML Models'),
        mpatches.Patch(facecolor='#E8F5E9', edgecolor='#388E3C', label='AI Advisor'),
        mpatches.Patch(facecolor='#FFF3E0', edgecolor='#F57C00', label='Trading'),
        mpatches.Patch(facecolor='#FCE4EC', edgecolor='#C2185B', label='UI'),
        mpatches.Patch(facecolor='#EFEBE9', edgecolor='#5D4037', label='Security'),
        mpatches.Patch(facecolor='#E0F2F1', edgecolor='#00695C', label='Database'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('class_diagram.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Class diagram saved as 'class_diagram.png'")

if __name__ == "__main__":
    create_class_diagram()
