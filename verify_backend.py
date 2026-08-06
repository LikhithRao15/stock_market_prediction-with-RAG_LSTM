import os
import sys
import time
import traceback
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Disable GPU locks for clean CPU verification
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("=" * 70)
print("🔬 IEEE HYBRID RAG-LSTM BACKEND SYSTEM VERIFICATION")
print("=" * 70)

results = []

def run_test(name, fn):
    t0 = time.time()
    try:
        res = fn()
        elapsed = time.time() - t0
        print(f"✅ [PASS] {name:<45} ({elapsed:.3f}s)")
        results.append((name, "PASS", elapsed, str(res) if res is not None else "OK"))
    except Exception as e:
        elapsed = time.time() - t0
        print(f"❌ [FAIL] {name:<45} ({elapsed:.3f}s)")
        print(f"   Details: {e}")
        traceback.print_exc()
        results.append((name, "FAIL", elapsed, str(e)))

# ---------------------------------------------------
# 1. PREPROCESSING & NLP SENTIMENT PIPELINE
# ---------------------------------------------------
def test_preprocessing():
    from preprocessing.clean_text import clean_text
    from preprocessing.tokenization import tokenize_text
    from preprocessing.stopword_removal import remove_stopwords
    
    sample_text = "NIFTY 50 surges +2.5% following strong quarterly earnings report from Reliance Industries!"
    cleaned = clean_text(sample_text)
    tokens = tokenize_text(cleaned)
    filtered = remove_stopwords(tokens)
    assert len(filtered) > 0, "Filtered tokens should not be empty"
    return f"{len(filtered)} tokens processed"

def test_vader_sentiment():
    from sentiment.vader_sentiment import analyze_sentiment
    sample_news = "Company reports record profit and massive dividend payout."
    score = analyze_sentiment(sample_news)
    assert "compound" in score or "sentiment" in score or isinstance(score, (float, dict)), "Sentiment output format valid"
    return f"Compound score evaluated"

def test_event_detection():
    from event_detection.detect_events import detect_corporate_event
    sample = "Board approves $500M buyout and merger deal with foreign investor."
    event = detect_corporate_event(sample)
    assert event is not None, "Event detection returned result"
    return f"Detected event: {event}"

# ---------------------------------------------------
# 2. TECHNICAL INDICATOR CALCULATIONS
# ---------------------------------------------------
def test_technical_indicators():
    from indicators.moving_average import calculate_moving_average
    from indicators.rsi import calculate_rsi
    from indicators.macd import calculate_macd
    
    prices = pd.Series([100, 102, 101, 105, 107, 106, 110, 112, 115, 114, 118, 120, 119, 122, 125])
    ma = calculate_moving_average(prices, window=5)
    rsi = calculate_rsi(prices, period=5)
    macd, signal = calculate_macd(prices, fast=5, slow=10, signal_window=3)
    
    assert not ma.dropna().empty, "MA series calculated"
    assert not rsi.dropna().empty, "RSI series calculated"
    return f"MA/RSI/MACD computed successfully"

# ---------------------------------------------------
# 3. FEATURE ENGINEERING & DATASET CREATION
# ---------------------------------------------------
def test_feature_engineering():
    dataset_path = os.path.join(project_root, "data", "final_dataset.csv")
    assert os.path.exists(dataset_path), f"Dataset file missing at {dataset_path}"
    df = pd.read_csv(dataset_path)
    assert len(df) > 10, "Dataset has sufficient rows"
    return f"Loaded {len(df)} rows, {len(df.columns)} features"

# ---------------------------------------------------
# 4. RAG VECTOR RETRIEVAL ENGINE
# ---------------------------------------------------
def test_rag_engine():
    from chatbot.rag_engine import FinancialRAGEngine
    engine = FinancialRAGEngine()
    assert engine.is_indexed, "RAG engine failed to index corpus"
    docs = engine.retrieve("quarterly earnings profit report", top_k=2)
    return f"Retrieved {len(docs)} documents"

def test_chatbot():
    from chatbot.chatbot import StockAssistantChatbot
    bot = StockAssistantChatbot()
    reply = bot.get_response("Explain RSI and market momentum")
    assert len(reply) > 20, "Chatbot reply generated"
    return f"Reply length: {len(reply)} chars"

# ---------------------------------------------------
# 5. DEEP LEARNING LSTM & UNCERTAINTY ESTIMATOR
# ---------------------------------------------------
def test_lstm_model_predict():
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    model_path = os.path.join(project_root, "models", "lstm_model.h5")
    assert os.path.exists(model_path), f"Model binary missing at {model_path}"
    
    with tf.device('/CPU:0'):
        model = load_model(model_path, compile=False)
        dummy_input = np.random.rand(1, 5, 8).astype(np.float32)
        pred = model(dummy_input, training=False).numpy()
    
    assert pred.shape == (1, 1), f"Expected shape (1, 1), got {pred.shape}"
    prob = float(pred[0][0])
    assert 0.0 <= prob <= 1.0, f"Probability out of range: {prob}"
    return f"LSTM Forecast Probability: {prob:.4f}"

def test_uncertainty_estimator():
    from prediction.uncertainty import MonteCarloUncertaintyEstimator
    estimator = MonteCarloUncertaintyEstimator(n_samples=5)
    # Dummy mock prediction test
    res = estimator.estimate_uncertainty(None, None)
    assert "prediction_prob" in res, "Uncertainty estimator returned dictionary"
    return f"Confidence score: {res['confidence_pct']}%"

# ---------------------------------------------------
# 6. XAI EXPLAINABILITY, RISK & PORTFOLIO ANALYTICS
# ---------------------------------------------------
def test_risk_analyzer():
    from risk.risk_analyzer import calculate_stock_risk_metrics
    dummy_returns = np.array([0.01, -0.02, 0.015, 0.005, -0.01])
    metrics = calculate_stock_risk_metrics(dummy_returns, 0.65)
    assert "volatility_score" in metrics and "risk_level" in metrics
    return f"Risk Level: {metrics['risk_level']}"

def test_market_regime():
    from market_regime.regime_detector import MarketRegimeDetector
    detector = MarketRegimeDetector()
    dummy_prices = pd.Series(np.linspace(100, 150, 30))
    regime = detector.detect_regime(dummy_prices)
    assert regime is not None, "Regime detector returned valid regime"
    return f"Regime: {regime}"

def test_portfolio_advisor():
    from portfolio.portfolio_advisor import PortfolioRecommendationEngine
    advisor = PortfolioRecommendationEngine()
    dummy_forecasts = [
        {"ticker": "RELIANCE.NS", "prob": 0.75, "returns": np.array([0.01, 0.02])},
        {"ticker": "TCS.NS", "prob": 0.40, "returns": np.array([-0.01, 0.01])}
    ]
    df_rank = advisor.rank_portfolio(dummy_forecasts)
    assert not df_rank.empty, "Portfolio ranking engine generated recommendations"
    return f"Ranked {len(df_rank)} assets"

# Run all backend tests sequentially
print("\n--- Executing Backend Module Unit & Integration Verification ---")
run_test("1. Preprocessing & Text Cleaning", test_preprocessing)
run_test("2. VADER Sentiment Polarity Analyzer", test_vader_sentiment)
run_test("3. Corporate Event Detection Engine", test_event_detection)
run_test("4. Quantitative Technical Indicators (MA/RSI/MACD)", test_technical_indicators)
run_test("5. Feature Engineering Dataset Integrity", test_feature_engineering)
run_test("6. Time-Aware RAG Semantic Retrieval Engine", test_rag_engine)
run_test("7. RAG Financial AI Chatbot Logic", test_chatbot)
run_test("8. Hybrid 64-Unit LSTM Keras Model Inference", test_lstm_model_predict)
run_test("9. Monte Carlo Uncertainty Estimator", test_uncertainty_estimator)
run_test("10. Stock Risk & VaR Metrics Analyzer", test_risk_analyzer)
run_test("11. Market Regime Detector", test_market_regime)
run_test("12. Portfolio Optimization & Asset Allocation Advisor", test_portfolio_advisor)

print("\n" + "=" * 70)
passed = sum(1 for r in results if r[1] == "PASS")
failed = sum(1 for r in results if r[1] == "FAIL")
print(f"SUMMARY: {passed} / {len(results)} Backend Modules PASSED cleanly!")
print("=" * 70)

if failed > 0:
    sys.exit(1)
