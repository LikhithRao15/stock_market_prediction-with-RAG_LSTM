# 📈 IEEE Research-Level Time-Aware Hybrid RAG-LSTM Stock Prediction System

## 1. Executive Summary

The **AI Stock Market Prediction System** is an end-to-end, IEEE research-level **Time-Aware Hybrid RAG-LSTM Application** designed to forecast short-term price movements (**UP 📈** or **DOWN 📉**) for National Stock Exchange of India (NSE) equities.

The system integrates:
1. **Semantic Vector RAG Engine**: SentenceTransformers (`all-MiniLM-L6-v2`) 384-dimensional dense semantic embeddings indexed in a high-performance **FAISS** vector store (`IndexFlatIP`).
2. **Strict Time-Aware Retrieval**: Enforces a non-negotiable temporal constraint ($\text{document\_date} \le T$) before prediction time $T$, completely eliminating future data leakage.
3. **Quantitative RAG Feature Extraction**: Computes 6 dynamic text-derived features (`RAG_Sentiment`, `RAG_Relevance`, `RAG_Event_Importance`, `RAG_Market_Impact`, `RAG_Risk_Score`, `RAG_Confidence`) to feed directly into the prediction model.
4. **Deep Learning (LSTM Neural Network)**: A 64-unit Long Short-Term Memory network trained on 5-step lookback feature sequences of combined Technical Indicators + RAG Semantic Features.
5. **Intraday Single-Interval Forecast Engine**: Single-interval forecasting focusing on **⚡ Next 5 Minutes (+5m)** alongside 15m, 30m, 60m, and daily sessions.
6. **Research Benchmark Suite**: Comprehensive 4-model comparative evaluation framework evaluating **LSTM Only**, **LSTM + Sentiment**, **LSTM + Event**, and **Hybrid Time-Aware RAG-LSTM**.
7. **Interactive Streamlit UI & AI Financial Chatbot**: Modern glassmorphic web dashboard with live Plotly candlestick subplots and an embedded RAG financial market assistant.

---

## 2. Directory Structure & Architecture Overview

```
stock_market_prediction/
├── preprocessing/          # Natural Language Processing Pipeline
│   ├── clean_text.py       # Text sanitization & regex cleaning
│   ├── tokenization.py     # NLTK tokenization with regex fallback & SSL bypass
│   └── stopword_removal.py # Stopword removal with NLTK & dictionary fallback
├── sentiment/              # VADER Sentiment Analysis
│   └── vader_sentiment.py  # Polarity scoring (-1.0 to +1.0)
├── event_detection/        # Corporate Event Classifier
│   └── detect_events.py    # Classifies events (Earnings, Merger, Legal, General)
├── indicators/             # Quantitative Technical Indicators
│   ├── moving_average.py   # 20-Day Simple Moving Average (MA_20)
│   ├── rsi.py              # Relative Strength Index (RSI - 14 period)
│   └── macd.py             # Moving Average Convergence Divergence (MACD)
├── feature_engineering/    # Feature Scaling & Dataset Construction
│   └── create_features.py  # Merges technicals, sentiment, events & RAG features
├── prediction/             # Deep Learning Model Training & Research Evaluation
│   ├── train_lstm.py       # Sequential Hybrid RAG-LSTM model training
│   ├── evaluate_models.py  # Benchmark suite comparing 4 model variants across 7 metrics
│   └── predict.py          # Standalone inference script for test sequences
├── chatbot/                # Semantic RAG Engine & Financial AI Assistant
│   ├── rag_engine.py       # SentenceTransformers & FAISS Time-Aware RAG Engine
│   └── chatbot.py          # RAG-augmented financial chatbot combining technicals & news retrieval
├── utils/                  # Helper Utilities & Path Resolution
│   └── helper.py           # Path helpers, dataset loaders & sequence scalers
├── webapp/                 # Streamlit Web Application & Dashboard
│   └── app.py              # Single 5m interval predictor, Plotly UI, RAG chatbot tab
├── models/                 # Model Artifact Storage
│   └── lstm_model.h5       # Saved Keras LSTM model binary
├── outputs/                # Evaluation & Benchmark Outputs
│   └── evaluation_results.csv # Exported benchmark evaluation metrics
├── data/                   # Data Storage Directory
│   ├── news_data.csv       # Raw news dataset
│   ├── news_processed.csv  # Processed news dataset with sentiment & events
│   ├── stock_data.csv      # Raw stock price dataset
│   ├── stock_with_indicators.csv # Stock dataset with RSI, MACD, MA_20
│   └── final_dataset.csv   # Unified dataset used for model training
└── process_news.py         # End-to-end news preprocessing pipeline runner
```

---

## 3. Detailed Component & Pipeline Breakdown

### Component A: Time-Aware Semantic RAG Engine
- **Dense Embedding Model**: Uses `SentenceTransformer('all-MiniLM-L6-v2')` to generate 384-dimensional dense semantic vectors.
- **FAISS Vector Database**: Indexes vectors in `faiss.IndexFlatIP` (normalized L2 inner product for exact Cosine Similarity).
- **Time-Aware Strict Constraint**: Ensures that for any target forecast at time $T$, candidate documents must satisfy:
  $$\text{published\_date} \le T$$
- **Quantitative RAG Features Output**:
  1. `RAG_Sentiment`: Weighted sentiment score of retrieved news.
  2. `RAG_Relevance`: Peak semantic similarity score.
  3. `RAG_Event_Importance`: Weighted importance of corporate events (`Earnings`: 0.90, `Merger`: 0.85, `Legal`: 0.80, `General`: 0.40).
  4. `RAG_Market_Impact`: $\text{Relevance} \times |\text{Sentiment}| \times \text{Event Importance}$.
  5. `RAG_Risk_Score`: High negative sentiment and legal event severity score.
  6. `RAG_Confidence`: Retrieval vector density confidence.

---

### Component B: Hybrid Feature Matrix (16 Features)
The LSTM model processes a 16-feature vector combining technical indicators and time-aware RAG features:

$$\text{Features} = [\text{RSI}, \text{MACD}, \text{Return}, \text{MA\_20\_ratio}, \text{Close\_Open}, \text{High\_Low}, \text{Volume\_ratio}, \text{Momentum\_Factor}, \text{Sentiment}, \text{Event}, \text{RAG\_Sentiment}, \text{RAG\_Relevance}, \text{RAG\_Event\_Importance}, \text{RAG\_Market\_Impact}, \text{RAG\_Risk\_Score}, \text{RAG\_Confidence}]$$

---

### Component C: Research Benchmark Suite ([prediction/evaluate_models.py](file:///Users/likhithraok/Desktop/stock_market_prediction/prediction/evaluate_models.py))
Systematically evaluates 4 comparative models on an identical test split:
1. **LSTM Only (Technical Indicators)**
2. **LSTM + VADER Sentiment**
3. **LSTM + Event Detection**
4. **Proposed: Hybrid Time-Aware RAG-LSTM**

Metrics reported:
- Accuracy, Precision, Recall, F1 Score, ROC-AUC, Confusion Matrix, and Inference Latency (ms/sample).

---

## 4. How to Run & Verify the System

### 1. Run Feature Engineering (Generate Time-Aware RAG Features)
```bash
./venv/bin/python feature_engineering/create_features.py
```

### 2. Train Hybrid RAG-LSTM Model
```bash
./venv/bin/python prediction/train_lstm.py
```

### 3. Execute Research Evaluation Benchmark
```bash
./venv/bin/python prediction/evaluate_models.py
```

### 4. Launch Streamlit Web Application
```bash
./venv/bin/streamlit run webapp/app.py
```
