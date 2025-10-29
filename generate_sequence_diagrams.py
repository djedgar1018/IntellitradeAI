"""
Generate Sequence Diagrams for IntelliTradeAI
Creates professional UML-style sequence diagrams as PNG images
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
import os

# Create diagrams directory
os.makedirs('diagrams', exist_ok=True)

class SequenceDiagram:
    def __init__(self, title, width=14, height=10):
        self.fig, self.ax = plt.subplots(figsize=(width, height))
        self.ax.set_xlim(0, width)
        self.ax.set_ylim(0, height)
        self.ax.axis('off')
        self.title = title
        self.actors = []
        self.current_y = height - 1.5
        
    def add_title(self):
        self.ax.text(self.fig.get_figwidth()/2, self.fig.get_figheight() - 0.5, 
                    self.title, ha='center', fontsize=16, fontweight='bold')
    
    def add_actor(self, x, name, is_system=False):
        """Add an actor/component box at the top"""
        color = 'lightblue' if not is_system else 'lightgray'
        box = FancyBboxPatch((x-0.8, self.current_y + 0.8), 1.6, 0.6,
                            boxstyle="round,pad=0.05", 
                            edgecolor='black', facecolor=color, linewidth=2)
        self.ax.add_patch(box)
        self.ax.text(x, self.current_y + 1.1, name, ha='center', va='center', 
                    fontsize=9, fontweight='bold', wrap=True)
        
        # Add lifeline
        self.ax.plot([x, x], [self.current_y + 0.8, 0.5], 'k--', linewidth=1)
        
        self.actors.append({'x': x, 'name': name})
        return x
    
    def add_message(self, from_x, to_x, message, y_offset, is_return=False, is_async=False):
        """Add a message arrow between actors"""
        y = self.current_y - y_offset
        
        # Arrow style
        if is_return:
            style = '<-'
            linestyle = 'dashed'
        else:
            style = '->'
            linestyle = 'solid'
        
        arrow = FancyArrowPatch((from_x, y), (to_x, y),
                               arrowstyle=style, mutation_scale=20,
                               linestyle=linestyle, linewidth=1.5, color='black')
        self.ax.add_patch(arrow)
        
        # Message label
        mid_x = (from_x + to_x) / 2
        label_y = y + 0.15
        self.ax.text(mid_x, label_y, message, ha='center', fontsize=8, 
                    style='italic' if is_return else 'normal',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none'))
        
        return y
    
    def add_activation(self, x, y_start, y_end):
        """Add activation box (execution specification)"""
        height = y_start - y_end
        box = Rectangle((x-0.1, y_end), 0.2, height, 
                       facecolor='white', edgecolor='black', linewidth=2)
        self.ax.add_patch(box)
    
    def add_note(self, x, y, text, width=2):
        """Add a note annotation"""
        box = FancyBboxPatch((x-width/2, y-0.3), width, 0.6,
                            boxstyle="round,pad=0.05", 
                            edgecolor='orange', facecolor='lightyellow', linewidth=2)
        self.ax.add_patch(box)
        self.ax.text(x, y, text, ha='center', va='center', fontsize=7, wrap=True)
    
    def save(self, filename):
        """Save the diagram"""
        plt.tight_layout()
        plt.savefig(f'diagrams/{filename}', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Created: diagrams/{filename}")

# ============================================================================
# DIAGRAM 1: Day Trader Getting Real-time Prediction
# ============================================================================
def create_day_trader_prediction_sequence():
    diagram = SequenceDiagram('Sequence Diagram: Day Trader Getting Real-time Prediction', width=14, height=11)
    # Title removed to prevent covering actor names
    
    # Actors
    user_x = diagram.add_actor(2, 'Day Trader\n(Web UI)', False)
    dashboard_x = diagram.add_actor(4.5, 'Streamlit\nDashboard', True)
    api_x = diagram.add_actor(7, 'FastAPI\nBackend', True)
    data_x = diagram.add_actor(9.5, 'Data\nIngestion', True)
    model_x = diagram.add_actor(12, 'ML Model\nPredictor', True)
    
    y_offset = 0.5
    
    # Flow
    diagram.add_message(user_x, dashboard_x, '1. Select asset (BTC)', y_offset, False)
    y_offset += 0.5
    
    diagram.add_message(user_x, dashboard_x, '2. Click "Get Prediction"', y_offset, False)
    y_offset += 0.5
    
    diagram.add_message(dashboard_x, api_x, '3. GET /predict?symbol=BTC', y_offset, False)
    y_offset += 0.5
    
    diagram.add_note(api_x, diagram.current_y - y_offset - 0.4, 'Check cache for\nrecent prediction', 1.8)
    y_offset += 1.0
    
    diagram.add_message(api_x, data_x, '4. fetch_market_data(BTC)', y_offset, False)
    y_offset += 0.5
    
    diagram.add_message(data_x, data_x + 0.01, '5. Call Yahoo Finance API', y_offset, False)
    y_offset += 0.5
    
    diagram.add_message(data_x, api_x, '6. Return OHLCV data', y_offset, True)
    y_offset += 0.5
    
    diagram.add_note(api_x, diagram.current_y - y_offset - 0.4, 'Calculate 50+\ntechnical indicators', 1.8)
    y_offset += 1.0
    
    diagram.add_message(api_x, model_x, '7. predict(features)', y_offset, False)
    y_offset += 0.5
    
    diagram.add_message(model_x, model_x + 0.01, '8. Load cached model', y_offset, False)
    y_offset += 0.5
    
    diagram.add_message(model_x, model_x + 0.01, '9. Ensemble prediction\n(RF+XGB+LSTM)', y_offset, False)
    y_offset += 0.5
    
    diagram.add_message(api_x, model_x, '10. Return {signal: BUY,\nconfidence: 85%}', y_offset, True)
    y_offset += 0.5
    
    diagram.add_message(dashboard_x, api_x, '11. JSON response', y_offset, True)
    y_offset += 0.5
    
    diagram.add_message(user_x, dashboard_x, '12. Display: BUY @ 85%\nwith price chart', y_offset, True)
    y_offset += 0.5
    
    diagram.add_note(user_x, diagram.current_y - y_offset - 0.5, 'Decision: Execute trade\nbased on signal', 2)
    
    diagram.save('seq_01_day_trader_prediction.png')

# ============================================================================
# DIAGRAM 2: Data Scientist Training ML Model
# ============================================================================
def create_model_training_sequence():
    diagram = SequenceDiagram('Sequence Diagram: Data Scientist Training ML Model', width=14, height=12)
    # Title removed to prevent covering actor names
    
    # Actors
    user_x = diagram.add_actor(2, 'Data Scientist\n(Web UI)', False)
    dashboard_x = diagram.add_actor(4.5, 'Streamlit\nDashboard', True)
    api_x = diagram.add_actor(7, 'FastAPI\nBackend', True)
    trainer_x = diagram.add_actor(9.5, 'Model\nTrainer', True)
    storage_x = diagram.add_actor(12, 'Model\nCache', True)
    
    y_offset = 0.5
    
    # Flow
    diagram.add_message(user_x, dashboard_x, '1. Select asset: AAPL', y_offset, False)
    y_offset += 0.5
    
    diagram.add_message(user_x, dashboard_x, '2. Choose: XGBoost', y_offset, False)
    y_offset += 0.5
    
    diagram.add_message(user_x, dashboard_x, '3. Set params:\nlookback=60, epochs=100', y_offset, False)
    y_offset += 0.5
    
    diagram.add_message(dashboard_x, api_x, '4. POST /retrain\n{symbol: AAPL, model: xgboost}', y_offset, False)
    y_offset += 0.5
    
    diagram.add_note(api_x, diagram.current_y - y_offset - 0.4, 'Validate parameters\nand check resources', 1.8)
    y_offset += 1.0
    
    diagram.add_message(api_x, trainer_x, '5. train_model(AAPL, xgboost)', y_offset, False)
    y_offset += 0.5
    
    diagram.add_message(trainer_x, trainer_x + 0.01, '6. Fetch 2 years\nhistorical data', y_offset, False)
    y_offset += 0.5
    
    diagram.add_message(trainer_x, trainer_x + 0.01, '7. Feature engineering\n(50+ indicators)', y_offset, False)
    y_offset += 0.5
    
    diagram.add_message(trainer_x, trainer_x + 0.01, '8. Train/test split 80/20', y_offset, False)
    y_offset += 0.5
    
    diagram.add_message(trainer_x, dashboard_x, '9. Progress update: 25%', y_offset, True)
    y_offset += 0.5
    
    diagram.add_message(trainer_x, trainer_x + 0.01, '10. XGBoost training\n(100 iterations)', y_offset, False)
    y_offset += 0.5
    
    diagram.add_message(trainer_x, dashboard_x, '11. Progress update: 75%', y_offset, True)
    y_offset += 0.5
    
    diagram.add_message(trainer_x, trainer_x + 0.01, '12. Evaluate on test set', y_offset, False)
    y_offset += 0.5
    
    diagram.add_message(trainer_x, storage_x, '13. Save model\nAAPL_xgboost.joblib', y_offset, False)
    y_offset += 0.5
    
    diagram.add_message(storage_x, trainer_x, '14. Confirm saved', y_offset, True)
    y_offset += 0.5
    
    diagram.add_message(trainer_x, api_x, '15. Return metrics:\naccuracy=82%, F1=0.79', y_offset, True)
    y_offset += 0.5
    
    diagram.add_message(api_x, dashboard_x, '16. Training complete\nwith results', y_offset, True)
    y_offset += 0.5
    
    diagram.add_message(dashboard_x, user_x, '17. Display accuracy chart\nand confusion matrix', y_offset, True)
    
    diagram.save('seq_02_model_training.png')

# ============================================================================
# DIAGRAM 3: Algorithm Developer Using API
# ============================================================================
def create_api_integration_sequence():
    diagram = SequenceDiagram('Sequence Diagram: Algorithm Developer Using REST API', width=14, height=10)
    # Title removed to prevent covering actor names
    
    # Actors
    user_x = diagram.add_actor(2, 'Trading Bot\n(Python Script)', False)
    api_x = diagram.add_actor(5, 'FastAPI\nBackend', True)
    auth_x = diagram.add_actor(7.5, 'Auth\nMiddleware', True)
    predictor_x = diagram.add_actor(10, 'Prediction\nEngine', True)
    webhook_x = diagram.add_actor(12.5, 'External\nBroker API', True)
    
    y_offset = 0.5
    
    # Flow
    diagram.add_message(user_x, api_x, '1. POST /api/predict\nHeader: API-Key', y_offset, False)
    y_offset += 0.5
    
    diagram.add_message(api_x, auth_x, '2. validate_api_key()', y_offset, False)
    y_offset += 0.5
    
    diagram.add_note(auth_x, diagram.current_y - y_offset - 0.4, 'Check rate limits\nand permissions', 1.5)
    y_offset += 1.0
    
    diagram.add_message(auth_x, api_x, '3. Authentication OK', y_offset, True)
    y_offset += 0.5
    
    diagram.add_message(api_x, predictor_x, '4. get_predictions(["BTC", "ETH"])', y_offset, False)
    y_offset += 0.5
    
    diagram.add_message(predictor_x, predictor_x + 0.01, '5. Process batch\n(parallel)', y_offset, False)
    y_offset += 0.5
    
    diagram.add_message(predictor_x, api_x, '6. Return predictions array', y_offset, True)
    y_offset += 0.5
    
    diagram.add_message(api_x, user_x, '7. JSON: [{symbol: BTC,\nsignal: BUY, conf: 85%}, ...]', y_offset, True)
    y_offset += 0.5
    
    diagram.add_note(user_x, diagram.current_y - y_offset - 0.4, 'Filter: confidence > 80%\nFound: BTC BUY signal', 1.8)
    y_offset += 1.0
    
    diagram.add_message(user_x, webhook_x, '8. POST /execute_trade\n{symbol: BTC, action: BUY}', y_offset, False)
    y_offset += 0.5
    
    diagram.add_message(webhook_x, user_x, '9. Order executed:\nID #12345', y_offset, True)
    y_offset += 0.5
    
    diagram.add_message(user_x, api_x, '10. POST /log_trade\n{symbol: BTC, entry: $45000}', y_offset, False)
    y_offset += 0.5
    
    diagram.add_message(api_x, user_x, '11. Trade logged', y_offset, True)
    
    diagram.save('seq_03_api_integration.png')

# ============================================================================
# DIAGRAM 4: Portfolio Manager Running Backtest
# ============================================================================
def create_backtest_sequence():
    diagram = SequenceDiagram('Sequence Diagram: Portfolio Manager Running Backtest', width=14, height=11)
    # Title removed to prevent covering actor names
    
    # Actors
    user_x = diagram.add_actor(2, 'Portfolio\nManager', False)
    dashboard_x = diagram.add_actor(4.5, 'Streamlit\nDashboard', True)
    api_x = diagram.add_actor(7, 'FastAPI\nBackend', True)
    backtest_x = diagram.add_actor(9.5, 'Backtesting\nEngine', True)
    db_x = diagram.add_actor(12, 'Historical\nData Store', True)
    
    y_offset = 0.5
    
    # Flow
    diagram.add_message(user_x, dashboard_x, '1. Select: ETH, 1-year', y_offset, False)
    y_offset += 0.5
    
    diagram.add_message(user_x, dashboard_x, '2. Set: $10,000 capital,\nstop-loss 2%', y_offset, False)
    y_offset += 0.5
    
    diagram.add_message(dashboard_x, api_x, '3. POST /backtest\n{params...}', y_offset, False)
    y_offset += 0.5
    
    diagram.add_message(api_x, backtest_x, '4. run_backtest(ETH, 365d)', y_offset, False)
    y_offset += 0.5
    
    diagram.add_message(backtest_x, db_x, '5. fetch_historical_data()', y_offset, False)
    y_offset += 0.5
    
    diagram.add_message(db_x, backtest_x, '6. Return 365 days OHLCV', y_offset, True)
    y_offset += 0.5
    
    diagram.add_note(backtest_x, diagram.current_y - y_offset - 0.4, 'Simulate trading\nday-by-day', 1.5)
    y_offset += 1.0
    
    diagram.add_message(backtest_x, backtest_x + 0.01, '7. For each day:\npredict → trade → update P&L', y_offset, False)
    y_offset += 0.5
    
    diagram.add_message(backtest_x, dashboard_x, '8. Progress: 50%\n(180 days simulated)', y_offset, True)
    y_offset += 0.5
    
    diagram.add_message(backtest_x, backtest_x + 0.01, '9. Continue simulation', y_offset, False)
    y_offset += 0.5
    
    diagram.add_message(backtest_x, backtest_x + 0.01, '10. Calculate metrics:\nSharpe, drawdown, etc.', y_offset, False)
    y_offset += 0.5
    
    diagram.add_message(backtest_x, api_x, '11. Return results:\nROI: +45%, Sharpe: 1.8', y_offset, True)
    y_offset += 0.5
    
    diagram.add_message(api_x, dashboard_x, '12. Backtest complete', y_offset, True)
    y_offset += 0.5
    
    diagram.add_message(dashboard_x, user_x, '13. Display equity curve,\ntrade log, metrics', y_offset, True)
    y_offset += 0.5
    
    diagram.add_note(user_x, diagram.current_y - y_offset - 0.5, 'Decision: Strategy validated\nProceed with live trading', 2)
    
    diagram.save('seq_04_backtest_analysis.png')

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    print("\n🎨 Generating Sequence Diagrams for IntelliTradeAI...\n")
    
    create_day_trader_prediction_sequence()
    create_model_training_sequence()
    create_api_integration_sequence()
    create_backtest_sequence()
    
    print("\n✅ All sequence diagrams generated successfully!")
    print("📁 Location: diagrams/")
    print("\nGenerated files:")
    print("  1. seq_01_day_trader_prediction.png")
    print("  2. seq_02_model_training.png")
    print("  3. seq_03_api_integration.png")
    print("  4. seq_04_backtest_analysis.png")
