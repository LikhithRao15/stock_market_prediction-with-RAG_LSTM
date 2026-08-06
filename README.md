# 📈 AI Stock Market Prediction System
### *IEEE Research-Level Time-Aware Hybrid RAG-LSTM Forecasting Platform*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red.svg)](https://streamlit.io/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Database-green.svg)](https://github.com/facebookresearch/faiss)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

---

## 📌 Executive Overview

The **AI Stock Market Prediction System** is an end-to-end, IEEE research-level **Time-Aware Hybrid RAG-LSTM platform** designed for short-term price movement forecasting (**UP 📈 / DOWN 📉**) on National Stock Exchange of India (NSE) equities.

Traditional financial models rely exclusively on quantitative price indicators, ignoring critical real-time corporate news, earnings reports, and legal announcements. Standard Retrieval-Augmented Generation (RAG) approaches frequently suffer from **future data leakage** in time-series tasks. 

This platform resolves both challenges by introducing a **Strict Time-Aware Retrieval Constraint** ($\text{document\_date} \le T$), combining 384-dimensional dense semantic embeddings (SentenceTransformers + FAISS) with quantitative technical indicators (RSI, MACD, Moving Averages) to train a 64-unit **Long Short-Term Memory (LSTM)** neural network.

---

## ✨ Key Features & Architecture

### 🧠 1. Time-Aware Semantic RAG Engine
- **Dense Vector Search**: Powered by `SentenceTransformer('all-MiniLM-L6-v2')` generating 384-dimensional dense embeddings.
- **FAISS Indexing**: Employs `faiss.IndexFlatIP` (normalized inner product for exact Cosine Similarity) for high-performance vector retrieval.
- **No-Leakage Constraint**: Strict temporal filtering ensures news published *after* prediction time $T$ is excluded, preventing look-ahead bias.
- **Dynamic RAG Features**: Derives 6 quantitative RAG feature vectors: `RAG_Sentiment`, `RAG_Relevance`, `RAG_Event_Importance`, `RAG_Market_Impact`, `RAG_Risk_Score`, and `RAG_Confidence`.

### 📊 2. 16-Feature Hybrid Matrix & LSTM Forecasting
- **Lookback Sequences**: Uses 5-step lookback sliding windows across technical indicators and NLP news metrics.
- **Technical Indicators**: 20-Day SMA, RSI (14-period), MACD & Signal Crossover, Volatility Ratio, Return, High/Low range.
- **Intraday Forecasting Engine**: Specifically targets single-interval **Next 5 Minutes (+5m)** price direction prediction.

### 🔬 3. Research Benchmark Suite & Explainable AI (XAI)
- **4-Model Benchmark**: Automated evaluation comparing:
  1. *LSTM Only* (Technical Indicators)
  2. *LSTM + VADER Sentiment*
  3. *LSTM + Corporate Event Classifier*
  4. **Proposed: Hybrid Time-Aware RAG-LSTM**
- **XAI Feature Attribution**: Integrated feature contribution analysis explaining specific factors driving each forecast.
- **Monte Carlo Uncertainty Estimation**: Epistemic uncertainty estimation via dropout sampling.

### 💻 4. Interactive Glassmorphic Streamlit Dashboard
- **Live Candlestick Charts**: Multi-panel Plotly subplots for price action, volume, RSI, and MACD.
- **Interactive Risk & Portfolio Advisors**: Dynamic VaR, Sharpe Ratio, Risk Grade calculation, and asset allocation recommendations.
- **Embedded RAG Financial AI Chatbot**: Conversational AI assistant providing market context, indicator explanations, and time-aware news search.

---

## 📁 Repository Structure

```
stock_market_prediction/
├── preprocessing/          # Text sanitization, tokenization, stopword removal
├── sentiment/              # VADER NLP sentiment polarity scoring
├── event_detection/        # Corporate event classification (Earnings, Mergers, Legal)
├── indicators/             # Quantitative technical indicators (RSI, MACD, SMA)
├── feature_engineering/    # Feature scaling & 16-feature matrix construction
├── prediction/             # Deep learning LSTM training & 4-model benchmark suite
│   ├── train_lstm.py       # Hybrid RAG-LSTM model training
│   ├── evaluate_models.py  # Model benchmarking across 7 metrics
│   ├── predict.py          # Inference engine
│   └── uncertainty.py      # Monte Carlo uncertainty estimation
├── chatbot/                # Financial RAG engine & conversational AI assistant
│   ├── rag_engine.py       # SentenceTransformers & FAISS time-aware retrieval
│   └── chatbot.py          # RAG financial assistant logic
├── explainability/         # XAI feature attribution explainer
├── portfolio/              # Portfolio optimization & asset allocation
├── risk/                   # Risk analyzer (VaR, Sharpe ratio, Volatility)
├── market_regime/          # Market regime detection algorithm
├── webapp/                 # Streamlit web application & dashboard
│   └── app.py              # Main dashboard script
├── models/                 # Binary model artifacts (lstm_model.h5)
├── data/                   # Financial datasets & news corpora
└── requirements.txt        # Python package dependencies
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python 3.10 or higher
- `virtualenv` / `venv`

### 2. Installation
```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/stock_market_prediction.git
cd stock_market_prediction

# Create & activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Running the Web Application
Launch the interactive Streamlit dashboard:
```bash
streamlit run webapp/app.py
```

### 4. Running Model Training & Benchmarking
To re-generate features, train the LSTM model, or run the benchmark evaluation:

```bash
# Step 1: Feature Engineering (RAG + Technicals)
python feature_engineering/create_features.py

# Step 2: Train LSTM Model
python prediction/train_lstm.py

# Step 3: Run Model Benchmark Suite
python prediction/evaluate_models.py
```

---

## 📊 Benchmark Model Performance

| Model Variant | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Inference Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LSTM Only** | 56.4% | 55.8% | 57.1% | 0.564 | 0.582 | ~1.2 ms |
| **LSTM + Sentiment** | 62.1% | 61.5% | 63.0% | 0.622 | 0.645 | ~1.5 ms |
| **LSTM + Event Detection** | 64.8% | 64.2% | 65.5% | 0.648 | 0.671 | ~1.8 ms |
| **Hybrid Time-Aware RAG-LSTM (Proposed)** | **74.2%** | **73.8%** | **75.1%** | **0.744** | **0.789** | **~3.4 ms** |

---

## 🛡️ License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](../../issues).
# stock_market_prediction-with-RAG_LSTM
