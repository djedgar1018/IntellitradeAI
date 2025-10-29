# IntelliTradeAI Sequence Diagrams

This folder contains professional UML sequence diagrams showing detailed interaction flows for key use cases in the IntelliTradeAI trading bot system.

## 📊 Diagram Overview

### 1. Day Trader Getting Real-time Prediction (`seq_01_day_trader_prediction.png`)
**Use Case**: Core trading prediction flow  
**Sprint**: Sprint 1  
**User Story**: Day Trader - Quick Asset Prediction

**Actors/Components**:
- Day Trader (Web UI) - User initiating the request
- Streamlit Dashboard - Frontend interface
- FastAPI Backend - API server orchestrating requests
- Data Ingestion - External API connector
- ML Model Predictor - AI prediction engine

**Flow Steps** (12 steps):
1. User selects asset (BTC)
2. User clicks "Get Prediction"
3. Dashboard sends GET request to API
4. API checks cache for recent predictions
5. API fetches current market data from Yahoo Finance
6. Data ingestion returns OHLCV data
7. API calculates 50+ technical indicators (RSI, MACD, etc.)
8. API sends features to ML Model
9. Model loads cached trained model
10. Model generates ensemble prediction (Random Forest + XGBoost + LSTM)
11. Model returns: `{signal: BUY, confidence: 85%}`
12. Dashboard displays BUY signal with confidence score and chart

**Key Notes**:
- Orange boxes show internal processing notes
- Dashed arrows indicate return messages
- Solid arrows show requests/calls
- Entire flow completes in ~2 seconds

**Performance**: Sub-3 second response time for real-time trading decisions

---

### 2. Data Scientist Training ML Model (`seq_02_model_training.png`)
**Use Case**: Model management and training  
**Sprint**: Sprint 2  
**User Story**: Data Scientist - Train Custom Models

**Actors/Components**:
- Data Scientist (Web UI) - ML expert configuring training
- Streamlit Dashboard - Training configuration interface
- FastAPI Backend - Training orchestrator
- Model Trainer - Core ML training logic
- Model Cache - Persistent model storage

**Flow Steps** (17 steps):
1. Scientist selects asset (AAPL)
2. Chooses algorithm (XGBoost)
3. Sets hyperparameters (lookback=60, epochs=100)
4. Dashboard sends POST /retrain request
5. API validates parameters and resources
6. Trainer fetches 2 years of historical data
7. Feature engineering creates 50+ technical indicators
8. Train/test split (80/20)
9. Progress update: 25% (sent to UI)
10. XGBoost training (100 iterations)
11. Progress update: 75%
12. Model evaluation on test set
13. Save model to cache (AAPL_xgboost.joblib)
14. Storage confirms save
15. Return metrics: accuracy=82%, F1=0.79
16. API sends completion notification
17. Dashboard displays accuracy chart and confusion matrix

**Key Notes**:
- Progress updates keep user informed during long training
- Model is cached for fast future predictions
- Metrics validate model quality before deployment

**Training Time**: 2-5 minutes depending on data size and algorithm

---

### 3. Algorithm Developer Using REST API (`seq_03_api_integration.png`)
**Use Case**: API integration for automated trading  
**Sprint**: Sprint 3  
**User Story**: Algorithm Developer - API Integration for Automation

**Actors/Components**:
- Trading Bot (Python Script) - Automated trading system
- FastAPI Backend - REST API server
- Auth Middleware - Security and rate limiting
- Prediction Engine - Batch prediction processor
- External Broker API - Trade execution platform

**Flow Steps** (11 steps):
1. Bot sends POST /api/predict with API key header
2. API forwards to authentication middleware
3. Middleware validates API key, checks rate limits
4. Auth returns "Authentication OK"
5. API calls prediction engine with batch: ["BTC", "ETH"]
6. Engine processes predictions in parallel
7. Engine returns predictions array
8. API sends JSON response with all predictions
9. Bot filters results (confidence > 80%, finds BTC BUY)
10. Bot executes trade via external broker API
11. Bot logs trade details back to IntelliTradeAI

**Key Notes**:
- API key authentication protects endpoints
- Batch processing enables efficient multi-asset screening
- Integration with external brokers enables full automation
- Trade logging maintains audit trail

**API Response Time**: <1 second for batch of 10 assets

---

### 4. Portfolio Manager Running Backtest (`seq_04_backtest_analysis.png`)
**Use Case**: Historical strategy validation  
**Sprint**: Sprint 2  
**User Story**: Portfolio Manager - Backtesting Engine

**Actors/Components**:
- Portfolio Manager - Professional trader validating strategy
- Streamlit Dashboard - Backtest configuration UI
- FastAPI Backend - Backtest orchestrator
- Backtesting Engine - Historical simulation engine
- Historical Data Store - Cached historical price data

**Flow Steps** (13 steps):
1. Manager selects asset (ETH) and timeframe (1 year)
2. Sets initial capital ($10,000) and stop-loss (2%)
3. Dashboard sends POST /backtest request
4. API initiates backtest run
5. Engine fetches historical data
6. Data store returns 365 days of OHLCV data
7. Engine simulates trading day-by-day (predict → trade → update P&L)
8. Progress update: 50% (180 days simulated)
9. Engine continues simulation
10. Engine calculates performance metrics (Sharpe ratio, drawdown)
11. Engine returns results: ROI: +45%, Sharpe: 1.8
12. API sends backtest complete notification
13. Dashboard displays equity curve, trade log, and metrics

**Key Notes**:
- Day-by-day simulation mirrors real trading
- Progress updates for long-running backtests
- Comprehensive metrics validate strategy quality
- Results inform go/no-go decision for live trading

**Backtest Duration**: 30-60 seconds for 1-year daily data

---

## 🎨 UML Sequence Diagram Legend

### Visual Elements

**Boxes at Top** = Participants (actors or system components)
- Blue boxes: External users/systems
- Gray boxes: Internal system components
- Dashed vertical lines: Lifelines showing time progression

**Solid Arrows** = Synchronous messages/requests
- Direction shows sender → receiver
- Label describes the action/message

**Dashed Arrows** = Return messages/responses
- Arrow points back from receiver → sender
- Label shows returned data

**Yellow/Orange Notes** = Important processing steps
- Annotations explaining internal logic
- Highlight key decisions or calculations

**White Rectangles on Lifelines** = Activation boxes
- Show when component is actively processing

---

## 📖 Mapping to User Stories

### Sprint 1: Core Trading Functionality
✅ **Sequence Diagram 1** implements:
- User Story 1.1: Quick Asset Prediction
- User Story 1.4: Confidence Scoring Display

### Sprint 2: Model Management & Analytics
✅ **Sequence Diagram 2** implements:
- User Story 2.1: Train Custom Models
- User Story 2.2: Model Comparison Dashboard (partial)

✅ **Sequence Diagram 4** implements:
- User Story 2.3: Backtesting Engine
- User Story 2.5: Risk Metrics Calculator (partial)

### Sprint 3: API Integration & Automation
✅ **Sequence Diagram 3** implements:
- User Story 3.1: REST API Endpoints
- User Story 3.4: Batch Prediction API
- User Story 3.5: Portfolio Tracking & P&L (partial)

---

## 🔄 Common Patterns Across Diagrams

### 1. **Request-Response Pattern**
All diagrams follow:
```
User → Frontend → Backend → Service → Response chain
```

### 2. **Caching for Performance**
- Diagram 1: Cache recent predictions
- Diagram 2: Cache trained models
- Diagram 4: Cache historical data

### 3. **Progress Updates for Long Operations**
- Diagram 2: Training progress (25%, 75%)
- Diagram 4: Backtest progress (50%)

### 4. **Authentication & Security**
- Diagram 3: API key validation and rate limiting

### 5. **External System Integration**
- Diagram 1: Yahoo Finance API
- Diagram 3: External Broker API
- All diagrams: Separation of concerns

---

## 💡 Technical Insights from Diagrams

### Component Responsibilities

**Streamlit Dashboard**:
- User interaction and input validation
- Real-time UI updates (progress bars, charts)
- Display results with visualizations

**FastAPI Backend**:
- Request routing and orchestration
- Business logic coordination
- API endpoint management
- Response formatting

**Data Ingestion**:
- External API connections (Yahoo Finance, CoinMarketCap)
- Data caching and validation
- OHLCV data retrieval

**ML Model Components**:
- Feature engineering (technical indicators)
- Model training and evaluation
- Prediction generation (ensemble methods)
- Model persistence (joblib cache)

**Backtesting Engine**:
- Historical simulation
- Performance metric calculation
- Trade execution simulation

---

## 🚀 Performance Characteristics

| Operation | Diagram | Typical Duration | Bottleneck |
|-----------|---------|------------------|------------|
| Get Prediction | #1 | 1-3 seconds | External API call |
| Train Model | #2 | 2-5 minutes | ML training iterations |
| API Batch Predict | #3 | <1 second | Model inference |
| Run Backtest | #4 | 30-60 seconds | Historical data processing |

---

## 🔧 Implementation Notes

### Asynchronous Processing
- **Diagram 2 & 4**: Long operations use async processing
- Progress updates via WebSocket or polling
- Non-blocking user interface

### Error Handling (Not Shown)
Each diagram implies error handling:
- API authentication failures
- External API timeouts
- Model training failures
- Invalid parameters

### Scalability Considerations
- **Diagram 3**: Batch processing enables horizontal scaling
- **Diagram 1**: Caching reduces redundant API calls
- **Diagram 2**: Model cache prevents redundant training

---

## 📚 Related Documentation

- **Use Case Diagrams**: High-level actor interactions
- **USER_MANUAL.md**: Detailed feature instructions
- **DEVELOPMENT_ROADMAP.md**: Sprint planning and timelines
- **PROJECT_CHANGES_ANALYSIS.md**: Technical architecture details

---

## 🔄 Diagram Maintenance

These diagrams were generated using Python with matplotlib. To regenerate or modify:

```bash
python generate_sequence_diagrams.py
```

**When to Update Diagrams**:
- New steps added to workflows
- External integrations change
- API endpoints modified
- Component interactions change

**Generator Script**: `generate_sequence_diagrams.py` in root directory

---

## 💼 Using These Diagrams

### For Technical Documentation
- Include in API documentation
- Use in developer onboarding
- Reference in code reviews

### For Stakeholder Presentations
- Explain system complexity and reliability
- Show integration capabilities
- Demonstrate performance characteristics

### For Testing & QA
- Create test cases from sequence steps
- Verify all paths are covered
- Test error scenarios not shown

### For Development Planning
- Identify integration points
- Estimate task complexity
- Plan API contracts

---

**Generated**: 2025-10-29  
**Tool**: IntelliTradeAI Sequence Diagram Generator  
**Format**: UML 2.5 Sequence Diagrams  
**Resolution**: 300 DPI PNG images  
**Coverage**: 4 core workflows across 3 sprints
