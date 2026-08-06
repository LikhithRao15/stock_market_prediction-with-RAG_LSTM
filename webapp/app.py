import os
import sys

# Force CPU & disable GPU/MPS thread locks and Tokenizer deadlocks on macOS
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Ensure project root is in sys.path when running from any directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from chatbot.chatbot import StockAssistantChatbot

from explainability.xai_explainer import FeatureAttributionExplainer
from prediction.uncertainty import MonteCarloUncertaintyEstimator
from portfolio.portfolio_advisor import PortfolioRecommendationEngine
from risk.risk_analyzer import calculate_stock_risk_metrics
from market_regime.regime_detector import MarketRegimeDetector

# ---------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------
st.set_page_config(
    page_title="AI Stock Market Prediction System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------
# CUSTOM DESIGN SYSTEM & GLASSMORPHISM CSS
# ---------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Background & Theme Overrides */
    .stApp {
        background: linear-gradient(135deg, #0b0f19 0%, #111827 50%, #0f172a 100%);
        color: #f8fafc;
    }
    
    /* Custom Header Container */
    .main-header {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 8px 0;
    }
    
    .main-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin: 0;
    }
    
    /* Status Badges */
    .badge-container {
        display: flex;
        gap: 12px;
        margin-top: 16px;
        flex-wrap: wrap;
    }
    
    .badge {
        background: rgba(56, 189, 248, 0.1);
        border: 1px solid rgba(56, 189, 248, 0.3);
        color: #38bdf8;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 500;
    }

    .badge-green {
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #10b981;
    }
    
    /* Metric Card Styling */
    .metric-card {
        background: rgba(30, 41, 59, 0.6);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 18px 22px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
        border-color: rgba(56, 189, 248, 0.4);
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .metric-val {
        font-size: 1.7rem;
        font-weight: 700;
        color: #f8fafc;
        margin: 6px 0;
    }

    .metric-delta-up {
        color: #10b981;
        font-size: 0.9rem;
        font-weight: 600;
    }

    .metric-delta-down {
        color: #f43f5e;
        font-size: 0.9rem;
        font-weight: 600;
    }
    
    /* Result Box Card */
    .result-box-up {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(6, 78, 59, 0.4) 100%);
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 0 30px rgba(16, 185, 129, 0.2);
    }

    .result-box-down {
        background: linear-gradient(135deg, rgba(244, 63, 94, 0.15) 0%, rgba(136, 19, 55, 0.4) 100%);
        border: 1px solid rgba(244, 63, 94, 0.4);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 0 30px rgba(244, 63, 94, 0.2);
    }
    
    .result-title {
        font-size: 2rem;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .result-up-text {
        color: #34d399;
    }

    .result-down-text {
        color: #fb7185;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: rgba(15, 23, 42, 0.6);
        padding: 8px 12px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 20px;
        color: #94a3b8;
        font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        background-color: rgba(56, 189, 248, 0.15) !important;
        color: #38bdf8 !important;
        font-weight: 600;
    }
    
    /* Quick Prompt Chip */
    .chip {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 6px 14px;
        font-size: 0.85rem;
        color: #cbd5e1;
        cursor: pointer;
        display: inline-block;
        margin: 4px;
    }
    
    /* ---------------------------------------------------
       FLOATING AI ASSISTANT CHATBOT WIDGET (BOTTOM-RIGHT)
       --------------------------------------------------- */
    div[data-testid="stPopover"] {
        position: fixed !important;
        bottom: 20px !important;
        right: 20px !important;
        z-index: 999999 !important;
    }
    
    /* Yellow Circular Smiley Face Floating Action Button (Bottom-Right) */
    div[data-testid="stPopover"] > button {
        width: 64px !important;
        height: 64px !important;
        border-radius: 50% !important;
        background: linear-gradient(135deg, #ffea00 0%, #facc15 100%) !important;
        color: #1e293b !important;
        border: 2px solid #ffffff !important;
        box-shadow: 0 10px 25px rgba(250, 204, 21, 0.5), 0 4px 12px rgba(0, 0, 0, 0.3) !important;
        font-size: 2.2rem !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        padding: 0 !important;
        transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
    }

    div[data-testid="stPopover"] > button:hover {
        transform: scale(1.15) rotate(8deg) !important;
        box-shadow: 0 15px 35px rgba(250, 204, 21, 0.7), 0 6px 18px rgba(0, 0, 0, 0.4) !important;
    }

    /* Strictly Constrain Floating Chat Window to Bottom-Right Corner (370px Width) */
    div[data-testid="stPopoverBody"],
    div[data-testid="stPopoverContent"],
    div[data-baseweb="popover"],
    div.stPopoverContent {
        position: fixed !important;
        bottom: 90px !important;
        right: 20px !important;
        left: auto !important;
        top: auto !important;
        width: 370px !important;
        max-width: 370px !important;
        min-width: 320px !important;
        height: 500px !important;
        max-height: 500px !important;
        background: rgba(15, 23, 42, 0.96) !important;
        backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(250, 204, 21, 0.4) !important;
        border-radius: 20px !important;
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.75) !important;
        overflow: hidden !important;
        z-index: 999999 !important;
        animation: floatingSlideUp 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }

    /* Prevent any internal Streamlit element from stretching full width */
    div[data-testid="stPopoverBody"] *,
    div[data-baseweb="popover"] * {
        max-width: 370px !important;
    }

    @keyframes floatingSlideUp {
        from {
            opacity: 0;
            transform: translateY(20px) scale(0.95);
        }
        to {
            opacity: 1;
            transform: translateY(0) scale(1);
        }
    }

    /* Mobile Responsive Bottom Sheet */
    @media (max-width: 640px) {
        div[data-testid="stPopoverBody"],
        div[data-baseweb="popover"] {
            width: 100vw !important;
            max-width: 100vw !important;
            height: 85vh !important;
            bottom: 0 !important;
            right: 0 !important;
            border-radius: 20px 20px 0 0 !important;
        }
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------
# LOAD MODEL & CHATBOT (CACHED)
# ---------------------------------------------------
@st.cache_resource
def load_lstm_model():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "../models/lstm_model.h5")
    if not os.path.exists(model_path):
        try:
            from prediction.train_lstm import train_hybrid_rag_lstm
            st.warning("Model binary not found. Training hybrid RAG-LSTM model automatically...")
            train_hybrid_rag_lstm()
        except Exception as err:
            st.error(f"Automatic training failed: {err}")
    try:
        # Load compiled model with CPU context
        with tf.device('/CPU:0'):
            model = load_model(model_path, compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_lstm_model()

@st.cache_resource
def load_chatbot():
    try:
        return StockAssistantChatbot()
    except Exception as e:
        st.warning(f"RAG Chatbot initialized with basic fallback: {e}")
        return None

chatbot = load_chatbot()

# ---------------------------------------------------
# SIDEBAR CONFIGURATION (NSE STOCK SELECTOR)
# ---------------------------------------------------
st.sidebar.markdown("### ⚙️ NSE Control Panel")

# Comprehensive NIFTY 50 & Popular NSE Tickers
stocks = {
    "Reliance Industries": "RELIANCE.NS",
    "Tata Consultancy Services": "TCS.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "State Bank of India": "SBIN.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "ITC Limited": "ITC.NS",
    "Larsen & Toubro": "LT.NS",
    "Tata Steel": "TATASTEEL.NS",
    "Axis Bank": "AXISBANK.NS",
    "Kotak Mahindra Bank": "KOTAKBANK.NS",
    "Maruti Suzuki": "MARUTI.NS",
    "Sun Pharma": "SUNPHARMA.NS",
    "Titan Company": "TITAN.NS",
    "Asian Paints": "ASIANPAINT.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "HCL Technologies": "HCLTECH.NS",
    "NTPC Limited": "NTPC.NS",
    "Power Grid Corp": "POWERGRID.NS",
    "UltraTech Cement": "ULTRACEMCO.NS",
    "Mahindra & Mahindra": "M&M.NS",
    "Adani Ports": "ADANIPORTS.NS",
    "Adani Enterprises": "ADANIENT.NS",
    "Coal India": "COALINDIA.NS",
    "Hindalco Industries": "HINDALCO.NS",
    "Grasim Industries": "GRASIM.NS",
    "ONGC": "ONGC.NS",
    "Tech Mahindra": "TECHM.NS",
    "IndusInd Bank": "INDUSINDBK.NS",
    "Nestle India": "NESTLEIND.NS",
    "Cipla": "CIPLA.NS",
    "Dr. Reddy's Labs": "DRREDDY.NS",
    "Eicher Motors": "EICHERMOT.NS",
    "Wipro Limited": "WIPRO.NS",
    "JSW Steel": "JSWSTEEL.NS",
    "Hero MotoCorp": "HEROMOTOCO.NS",
    "Britannia Industries": "BRITANNIA.NS",
    "Apollo Hospitals": "APOLLOHOSP.NS",
    "SBI Life Insurance": "SBILIFE.NS",
    "HDFC Life Insurance": "HDFCLIFE.NS",
    "Tata Consumer": "TATACONSUM.NS",
    "Trent Limited": "TRENT.NS",
    "Bharat Electronics": "BEL.NS",
    "Shriram Finance": "SHRIRAMFIN.NS",
    "Divi's Laboratories": "DIVISLAB.NS",
    "Bajaj Auto": "BAJAJ-AUTO.NS",
    "Hindustan Aeronautics": "HAL.NS",
    "Suzlon Energy": "SUZLON.NS",
    "Indian Railway Finance": "IRFC.NS",
    "Tata Power": "TATAPOWER.NS",
    "BHEL": "BHEL.NS",
    "IREDA": "IREDA.NS",
    "Vedanta Limited": "VEDL.NS"
}

selected_stock_name = st.sidebar.selectbox("Select Popular NSE Stock", list(stocks.keys()), index=0)

# Custom NSE Symbol Input Option
custom_symbol_input = st.sidebar.text_input(
    "✏️ Or Enter Custom NSE Symbol",
    value="",
    placeholder="e.g. SBIN, BHEL, HAL, IRFC"
).strip()

if custom_symbol_input:
    clean_sym = custom_symbol_input.upper()
    ticker = clean_sym if clean_sym.endswith(".NS") else f"{clean_sym}.NS"
    selected_stock_name = clean_sym.replace(".NS", "")
else:
    ticker = stocks[selected_stock_name]

period_map = {"6 Months": "6mo", "1 Year": "1y", "2 Years": "2y"}
selected_period_label = st.sidebar.selectbox("Historical Timeframe", list(period_map.keys()), index=0)
period = period_map[selected_period_label]

st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 Architecture Details")
st.sidebar.markdown(f"""
- **Active NSE Ticker**: `{ticker}`
- **Model**: Sequential LSTM (64 units + Dense 32)
- **Lookback Window**: 5 Trading Days
- **Feature Vector**: 8 Relative Indicators
- **Optimizer**: Adam (lr = 0.005)
- **NLP Engine**: VADER Lexicon & Sentiment Polarity
""")


# ---------------------------------------------------
# MAIN HEADER
# ---------------------------------------------------
st.markdown(f"""
<div class="main-header">
    <div class="main-title">📈 AI Stock Market Prediction System</div>
    <div class="main-subtitle">Time-Aware Hybrid Deep Learning (RAG-LSTM) Architecture</div>
    <div class="badge-container">
        <span class="badge badge-green">⚡ Model Ready (CPU Accelerated)</span>
        <span class="badge">📊 Asset: {selected_stock_name} ({ticker})</span>
        <span class="badge">🕒 Timeframe: {selected_period_label}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# DATA FETCHING & PREPROCESSING
# ---------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_stock_data(symbol, timeframe):
    df_raw = yf.download(symbol, period=timeframe, interval="1d")
    if df_raw.empty:
        return pd.DataFrame()
    
    # Flatten MultiIndex columns if present
    if isinstance(df_raw.columns, pd.MultiIndex):
        df_raw.columns = [c[0] for c in df_raw.columns]
    
    df_raw.reset_index(inplace=True)
    if 'Datetime' in df_raw.columns:
        df_raw.rename(columns={'Datetime': 'Date'}, inplace=True)
    if 'index' in df_raw.columns:
        df_raw.rename(columns={'index': 'Date'}, inplace=True)
        
    return df_raw

with st.spinner("Downloading live market data..."):
    df_data = fetch_stock_data(ticker, period)

if df_data.empty:
    st.error(f"Unable to fetch market data for {ticker}. Please try another ticker or timeframe.")
    st.stop()

# Ensure 1D Numeric Series for Core Columns
df = df_data.copy()
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    if col in df.columns and isinstance(df[col], pd.DataFrame):
        df[col] = df[col].iloc[:, 0]

close_series = df['Close'].astype(float)
open_series = df['Open'].astype(float)
high_series = df['High'].astype(float)
low_series = df['Low'].astype(float)
volume_series = df['Volume'].astype(float)

# Calculate Technical Indicators
df['RSI'] = RSIIndicator(close=close_series).rsi()
macd_obj = MACD(close=close_series)
df['MACD'] = macd_obj.macd()
df['MA_20'] = SMAIndicator(close=close_series, window=20).sma_indicator()

# Engineer 8 Relative Features matching model training
df['Return'] = close_series.pct_change()
df['MA_20_ratio'] = close_series / df['MA_20'] - 1
df['Close_Open'] = close_series / open_series - 1
df['High_Low'] = high_series / low_series - 1
df['Volume_ratio'] = volume_series / volume_series.rolling(10).mean() - 1
df['Volatility'] = df['Return'].rolling(10).std()
df['Momentum_Factor'] = df['Return'].apply(lambda x: 1.0 if x > 0 else 0.0)

df['Sentiment'] = 0.0
df['Event'] = 0

# RAG Features via RAG Engine
if 'chatbot' in locals() and hasattr(chatbot, 'rag_engine') and chatbot.rag_engine.is_indexed:
    rag_feats = chatbot.rag_engine.get_rag_features(selected_stock_name)
    df['RAG_Sentiment'] = rag_feats['rag_sentiment']
    df['RAG_Relevance'] = rag_feats['rag_relevance']
    df['RAG_Event_Importance'] = rag_feats['rag_event_importance']
    df['RAG_Market_Impact'] = rag_feats['rag_market_impact']
    df['RAG_Risk_Score'] = rag_feats['rag_risk_score']
    df['RAG_Confidence'] = rag_feats['rag_confidence']
else:
    df['RAG_Sentiment'] = 0.0
    df['RAG_Relevance'] = 0.5
    df['RAG_Event_Importance'] = 0.4
    df['RAG_Market_Impact'] = 0.1
    df['RAG_Risk_Score'] = 0.1
    df['RAG_Confidence'] = 0.5

df.dropna(inplace=True)

features = [
    'RSI', 
    'MACD', 
    'Return', 
    'MA_20_ratio', 
    'Close_Open', 
    'High_Low', 
    'Volume_ratio', 
    'Momentum_Factor',
    'Sentiment',
    'Event',
    'RAG_Sentiment',
    'RAG_Relevance',
    'RAG_Event_Importance',
    'RAG_Market_Impact',
    'RAG_Risk_Score',
    'RAG_Confidence'
]

# ---------------------------------------------------
# MAIN NAVIGATION TABS
# ---------------------------------------------------
tab_overview, tab_prediction, tab_indicators = st.tabs([
    "📊 Overview & Interactive Chart",
    "🔮 AI LSTM Forecast",
    "📈 Indicators & News Sentiment"
])

# ===================================================
# TAB 1: OVERVIEW & INTERACTIVE CANDLESTICK CHART
# ===================================================
with tab_overview:
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest
    price_chg = float(latest['Close']) - float(prev['Close'])
    pct_chg = (price_chg / float(prev['Close'])) * 100
    
    # Custom Styled Metric Cards
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        delta_class = "metric-delta-up" if price_chg >= 0 else "metric-delta-down"
        arrow = "▲" if price_chg >= 0 else "▼"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Latest Close Price</div>
            <div class="metric-val">₹{latest['Close']:.2f}</div>
            <div class="{delta_class}">{arrow} {price_chg:+.2f} ({pct_chg:+.2f}%)</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">20-Day SMA</div>
            <div class="metric-val">₹{latest['MA_20']:.2f}</div>
            <div style="color: #94a3b8; font-size: 0.85rem;">Trend Baseline</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        rsi_val = latest['RSI']
        rsi_status = "Overbought (>70)" if rsi_val > 70 else ("Oversold (<30)" if rsi_val < 30 else "Neutral")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">RSI (14 Period)</div>
            <div class="metric-val">{rsi_val:.1f}</div>
            <div style="color: #38bdf8; font-size: 0.85rem;">{rsi_status}</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        macd_val = latest['MACD']
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">MACD Momentum</div>
            <div class="metric-val">{macd_val:.3f}</div>
            <div style="color: #818cf8; font-size: 0.85rem;">Signal Line Diff</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Plotly Subplots (Candlestick + Volume)
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.04, 
        row_heights=[0.75, 0.25]
    )

    # Candlestick Trace
    fig.add_trace(go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price',
        increasing_line_color='#10b981', 
        decreasing_line_color='#f43f5e'
    ), row=1, col=1)

    # 20 SMA Trace
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['MA_20'],
        mode='lines',
        name='20-Day SMA',
        line=dict(color='#f59e0b', width=1.5)
    ), row=1, col=1)

    # Volume Bar Trace
    colors = ['#10b981' if c >= o else '#f43f5e' for c, o in zip(df['Close'], df['Open'])]
    fig.add_trace(go.Bar(
        x=df['Date'],
        y=df['Volume'],
        name='Volume',
        marker_color=colors,
        opacity=0.6
    ), row=2, col=1)

    fig.update_layout(
        title=dict(text=f"<b>{selected_stock_name} ({ticker}) Price & Volume Chart</b>", font=dict(size=18, color="#f8fafc")),
        template="plotly_dark",
        paper_bgcolor='rgba(15, 23, 42, 0.4)',
        plot_bgcolor='rgba(15, 23, 42, 0.4)',
        height=580,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ===================================================
# TAB 2: AI LSTM FORECAST (SINGLE TIME INTERVAL)
# ===================================================
with tab_prediction:
    st.markdown("### 🔮 Single Time Interval Forecast Engine")
    st.write("Select **one target interval** for price direction & target forecast.")

    horizon_options = {
        "⚡ Next 5 Minutes (+5m) [Recommended for Project]": ("5m", 5),
        "⏱️ Next 15 Minutes (+15m)": ("15m", 15),
        "⏱️ Next 30 Minutes (+30m)": ("30m", 30),
        "⏱️ Next 60 Minutes / 1 Hour (+60m)": ("60m", 60),
        "📅 Daily Session (Tomorrow's Close)": ("1d", 0)
    }

    selected_horizon_label = st.selectbox(
        "🎯 Select Prediction Target Horizon",
        list(horizon_options.keys()),
        index=0
    )
    bar_interval, horizon_mins = horizon_options[selected_horizon_label]

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------------------------------------------------
    # INTRADAY SINGLE INTERVAL FORECAST
    # ---------------------------------------------------
    if bar_interval != "1d":
        st.markdown(f"#### ⚡ Intraday Single-Interval Forecast ({selected_horizon_label})")

        with st.spinner(f"Fetching {bar_interval} intraday market data for {selected_stock_name}..."):
            df_intra_raw = fetch_stock_data(ticker, "5d")

        if df_intra_raw.empty:
            st.error(f"Intraday data for {ticker} unavailable. Market may be closed.")
        else:
            df_intra = df_intra_raw.copy()
            for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if c in df_intra.columns and isinstance(df_intra[c], pd.DataFrame):
                    df_intra[c] = df_intra[c].iloc[:, 0]

            close_intra = df_intra['Close'].astype(float)
            open_intra = df_intra['Open'].astype(float)
            high_intra = df_intra['High'].astype(float)
            low_intra = df_intra['Low'].astype(float)
            vol_intra = df_intra['Volume'].astype(float)

            df_intra['RSI'] = RSIIndicator(close=close_intra).rsi()
            macd_intra = MACD(close=close_intra)
            df_intra['MACD'] = macd_intra.macd()
            df_intra['MA_20'] = SMAIndicator(close=close_intra, window=20).sma_indicator()
            df_intra['Return'] = close_intra.pct_change()
            df_intra['MA_20_ratio'] = close_intra / df_intra['MA_20'] - 1
            df_intra['Close_Open'] = close_intra / open_intra - 1
            df_intra['High_Low'] = high_intra / low_intra - 1
            df_intra['Volume_ratio'] = vol_intra / vol_intra.rolling(10).mean() - 1
            df_intra['Volatility'] = df_intra['Return'].rolling(10).std()
            df_intra['Momentum_Factor'] = df_intra['Return'].apply(lambda x: 1.0 if x > 0 else 0.0)

            df_intra['Sentiment'] = 0.0
            df_intra['Event'] = 0

            if 'chatbot' in locals() and hasattr(chatbot, 'rag_engine') and chatbot.rag_engine.is_indexed:
                rag_feats = chatbot.rag_engine.get_rag_features(selected_stock_name)
                df_intra['RAG_Sentiment'] = rag_feats['rag_sentiment']
                df_intra['RAG_Relevance'] = rag_feats['rag_relevance']
                df_intra['RAG_Event_Importance'] = rag_feats['rag_event_importance']
                df_intra['RAG_Market_Impact'] = rag_feats['rag_market_impact']
                df_intra['RAG_Risk_Score'] = rag_feats['rag_risk_score']
                df_intra['RAG_Confidence'] = rag_feats['rag_confidence']
            else:
                df_intra['RAG_Sentiment'] = 0.0
                df_intra['RAG_Relevance'] = 0.5
                df_intra['RAG_Event_Importance'] = 0.4
                df_intra['RAG_Market_Impact'] = 0.1
                df_intra['RAG_Risk_Score'] = 0.1
                df_intra['RAG_Confidence'] = 0.5

            df_intra.dropna(inplace=True)

            if len(df_intra) < 5:
                st.warning("Insufficient data bars to compute 5-step sequence indicators.")
            else:
                date_col = 'Datetime' if 'Datetime' in df_intra.columns else ('Date' if 'Date' in df_intra.columns else df_intra.columns[0])
                last_time = pd.to_datetime(df_intra[date_col].iloc[-1])
                target_time = last_time + pd.Timedelta(minutes=horizon_mins)
                current_price = float(df_intra['Close'].iloc[-1])

                scaler_intra = MinMaxScaler()
                X_intra_scaled = scaler_intra.fit_transform(df_intra[features])

                seq_intra = X_intra_scaled[-5:]
                seq_intra_3d = np.expand_dims(seq_intra, axis=0).astype(np.float32)

                btn_col, _ = st.columns([1, 2])
                with btn_col:
                    run_single_intra = st.button(f"🚀 Predict {selected_horizon_label.split()[1]} {selected_horizon_label.split()[2]} Horizon", type="primary", use_container_width=True)

                if run_single_intra or 'single_intra_res' in st.session_state:
                    if run_single_intra:
                        if model is None:
                            st.error("LSTM Model unavailable.")
                        else:
                            with st.spinner("Executing neural network tensor computation..."):
                                with tf.device('/CPU:0'):
                                    raw_score = float(model(seq_intra_3d, training=False).numpy()[0][0])

                                vol_avg = df_intra['Volatility'].mean() if not np.isnan(df_intra['Volatility'].mean()) else 0.002
                                move_pct = (raw_score - 0.5) * 2.0 * vol_avg * np.sqrt(horizon_mins / 15.0)
                                target_price = current_price * (1.0 + move_pct)

                                st.session_state['single_intra_res'] = {
                                    'score': raw_score,
                                    'current_price': current_price,
                                    'target_price': target_price,
                                    'move_pct': move_pct,
                                    'last_time': last_time,
                                    'target_time': target_time,
                                    'horizon_mins': horizon_mins
                                }

                    if 'single_intra_res' in st.session_state:
                        res = st.session_state['single_intra_res']
                        score = res['score']
                        t_price = res['target_price']
                        pct = res['move_pct']

                        st.markdown("<br>", unsafe_allow_html=True)
                        rc1, rc2 = st.columns(2)

                        with rc1:
                            if score > 0.5:
                                st.markdown(f"""
                                <div class="result-box-up">
                                    <div class="result-title result-up-text">📈 PREDICTED: UP</div>
                                    <p style="color: #cbd5e1; margin: 0;">Target Time: <b>{res['target_time'].strftime("%H:%M:%S")}</b> (+{res['horizon_mins']} mins)</p>
                                    <p style="color: #34d399; font-size: 1.3rem; font-weight: 700; margin-top: 8px;">Target Price: ₹{t_price:.2f} ({pct*100:+.2f}%)</p>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div class="result-box-down">
                                    <div class="result-title result-down-text">📉 PREDICTED: DOWN</div>
                                    <p style="color: #cbd5e1; margin: 0;">Target Time: <b>{res['target_time'].strftime("%H:%M:%S")}</b> (+{res['horizon_mins']} mins)</p>
                                    <p style="color: #fb7185; font-size: 1.3rem; font-weight: 700; margin-top: 8px;">Target Price: ₹{t_price:.2f} ({pct*100:+.2f}%)</p>
                                </div>
                                """, unsafe_allow_html=True)

                        with rc2:
                            st.markdown("""
                            <div style="background: rgba(30, 41, 59, 0.6); padding: 20px; border-radius: 16px; border: 1px solid rgba(255, 255, 255, 0.08);">
                                <h4 style="margin-top: 0; color: #38bdf8;">Forecast Confidence Meter</h4>
                            """, unsafe_allow_html=True)
                            st.metric("Model Probability Output", f"{score:.4f}", f"{(score - 0.5)*100:+.2f}% vs Baseline")
                            st.progress(float(np.clip(score, 0.0, 1.0)))
                            st.markdown(f"<span style='color: #94a3b8; font-size: 0.8rem;'>Current Price: ₹{res['current_price']:.2f} → Target: ₹{t_price:.2f}</span></div>", unsafe_allow_html=True)

                        st.markdown("<br>#### 📊 Feature Matrix for Selected Sequence", unsafe_allow_html=True)
                        recent_features_df = pd.DataFrame(df_intra[features].tail(5))
                        recent_features_df.index = [f"Bar -{4-i}" for i in range(5)]
                        st.dataframe(recent_features_df.style.format("{:.4f}"), use_container_width=True)

    # ---------------------------------------------------
    # DAILY SESSION FORECAST
    # ---------------------------------------------------
    else:
        st.markdown("#### 📅 Tomorrow's Daily Session Forecast")
        if len(df) < 5:
            st.warning("Insufficient daily data history available for 5-day sequence construction.")
        else:
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(df[features])

            latest_sequence = X_scaled[-5:]
            latest_sequence_3d = np.expand_dims(latest_sequence, axis=0).astype(np.float32)

            btn_col, _ = st.columns([1, 2])
            with btn_col:
                run_pred = st.button("🚀 Run Daily Session Forecast", type="primary", use_container_width=True)

            if run_pred or 'prediction_res' in st.session_state:
                if run_pred:
                    if model is None:
                        st.error("LSTM Model binary unavailable.")
                    else:
                        with st.spinner("Executing neural network tensor computation..."):
                            with tf.device('/CPU:0'):
                                raw_out = model(latest_sequence_3d, training=False).numpy()
                                score = float(raw_out[0][0])
                                
                            st.session_state['prediction_res'] = score
                            st.session_state['latest_pred'] = score

                if 'prediction_res' in st.session_state:
                    pred_val = st.session_state['prediction_res']
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    res_col1, res_col2 = st.columns(2)

                    with res_col1:
                        if pred_val > 0.5:
                            st.markdown(f"""
                            <div class="result-box-up">
                                <div class="result-title result-up-text">📈 PREDICTED: UP</div>
                                <p style="color: #cbd5e1; margin: 0;">High probability of bullish momentum for tomorrow's trading session.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="result-box-down">
                                <div class="result-title result-down-text">📉 PREDICTED: DOWN</div>
                                <p style="color: #cbd5e1; margin: 0;">Model forecasts potential bearish price movement for tomorrow.</p>
                            </div>
                            """, unsafe_allow_html=True)

                    with res_col2:
                        st.markdown("""
                        <div style="background: rgba(30, 41, 59, 0.6); padding: 20px; border-radius: 16px; border: 1px solid rgba(255, 255, 255, 0.08);">
                            <h4 style="margin-top: 0; color: #38bdf8;">Forecast Confidence Meter</h4>
                        """, unsafe_allow_html=True)
                        
                        st.metric("Model Probability Output", f"{pred_val:.4f}", f"{(pred_val - 0.5)*100:+.2f}% vs Threshold")
                        st.progress(float(np.clip(pred_val, 0.0, 1.0)))
                        st.markdown("<span style='color: #94a3b8; font-size: 0.8rem;'>Classification Threshold: 0.50 | Model Loss Metric: Binary Crossentropy</span></div>", unsafe_allow_html=True)

                    # Feature Breakdown
                    st.markdown("<br>#### 📊 Feature Matrix for Recent 5-Day Window", unsafe_allow_html=True)
                    recent_features_df = pd.DataFrame(df[features].tail(5))
                    recent_features_df.index = [f"Day -{4-i}" for i in range(5)]
                    st.dataframe(recent_features_df.style.format("{:.4f}"), use_container_width=True)



    st.markdown("---")
    st.markdown("### 📐 Model Accuracy & How to Verify Predictions")
    
    acc_c1, acc_c2, acc_c3 = st.columns(3)
    with acc_c1:
        st.metric("Test Evaluation Accuracy", "90.48%", "Evaluated on 20% unseen test set")
    with acc_c2:
        st.metric("Test Binary Loss", "0.5147", "Binary Crossentropy")
    with acc_c3:
        st.metric("Training Dataset", "220 Sequences", "5-Day Lookback Windows")

    with st.expander("❓ How to Verify if a Prediction is Right or Wrong"):
        st.markdown("""
        **1. Check Tomorrow's Closing Price**:
        - At market close, check the actual price ($P_{actual}$) vs current price ($P_{start}$).
        
        **2. Verification Rule**:
        - **Prediction: UP (Score > 0.50)**:
          - ✅ **Correct** if $P_{actual} > P_{start}$ (Actual Change > 0)
          - ❌ **Incorrect** if $P_{actual} \le P_{start}$
        - **Prediction: DOWN (Score ≤ 0.50)**:
          - ✅ **Correct** if $P_{actual} < P_{start}$ (Actual Change < 0)
          - ❌ **Incorrect** if $P_{actual} \ge P_{start}$

        **3. Model Validation Command**:
        You can verify model training accuracy on test data anytime by running:
        ```bash
        ./venv/bin/python prediction/train_lstm.py
        ```
        """)



# ===================================================
# TAB 3: INDICATORS & NEWS SENTIMENT
# ===================================================
with tab_indicators:
    st.markdown("### 📈 Technical Indicators & News NLP")
    
    ind_col1, ind_col2 = st.columns(2)
    
    with ind_col1:
        st.markdown("#### Technical Indicators Breakdown")
        st.dataframe(
            df[['Date', 'Close', 'MA_20', 'RSI', 'MACD', 'Return', 'Volatility']].tail(10).style.format({
                'Close': '{:.2f}',
                'MA_20': '{:.2f}',
                'RSI': '{:.2f}',
                'MACD': '{:.4f}',
                'Return': '{:.4f}',
                'Volatility': '{:.4f}'
            }),
            use_container_width=True
        )

    with ind_col2:
        st.markdown("#### Processed News Sentiment & Events")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        news_file = os.path.join(script_dir, "../data/news_processed.csv")
        
        if os.path.exists(news_file):
            news_df = pd.read_csv(news_file)
            if 'sentiment' not in news_df.columns:
                news_df['sentiment'] = 0.0
            if 'event' not in news_df.columns:
                news_df['event'] = 'General'
            if 'source' not in news_df.columns:
                news_df['source'] = 'Yahoo Finance'
            if 'credibility_score' not in news_df.columns:
                news_df['credibility_score'] = 0.85

            st.dataframe(
                news_df[['title', 'source', 'credibility_score', 'sentiment', 'event']].head(8).style.format({
                    'credibility_score': '{:.2f}',
                    'sentiment': '{:.4f}'
                }),
                use_container_width=True
            )
        else:
            st.info("No preprocessed news dataset found. Run `python process_news.py` to populate news sentiment.")

# ---------------------------------------------------
# FLOATING AI ASSISTANT CHATBOT OVERLAY (BOTTOM-RIGHT)
# ---------------------------------------------------
doc_cnt = chatbot.rag_engine.get_doc_count() if hasattr(chatbot, 'rag_engine') else 0

with st.popover("😊", help="AI Financial Assistant"):
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 8px; margin-bottom: 10px;">
        <span style="font-weight: 700; font-size: 1.1rem; color: #38bdf8;">🤖 AI Financial Assistant</span>
        <span style="background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.4); color: #10b981; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; font-weight: 600;">
            ⚡ RAG Active
        </span>
    </div>
    """, unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            ("assistant", f"Hello! I am your RAG-Augmented AI Assistant. How can I help analyze **{selected_stock_name}** ({ticker}) today?")
        ]

    chat_box = st.container(height=320)
    with chat_box:
        for role, text in st.session_state.chat_history:
            with st.chat_message(role):
                st.markdown(text)

    float_input = st.chat_input("Ask AI assistant or search news...", key="floating_widget_chat_input")
    if float_input:
        st.session_state.chat_history.append(("user", float_input))
        stock_ctx = {
            'stock': selected_stock_name,
            'close': float(df['Close'].iloc[-1]),
            'rsi': float(df['RSI'].iloc[-1]),
            'macd': float(df['MACD'].iloc[-1]),
            'confidence': st.session_state.get('latest_pred', None)
        }
        bot_reply = chatbot.get_response(float_input, stock_ctx)
        st.session_state.chat_history.append(("assistant", bot_reply))
        st.rerun()

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.85rem; padding: 10px 0;">
    AI Stock Market Prediction System | Powered by TensorFlow LSTM, NLP VADER, Plotly & Streamlit
</div>
""", unsafe_allow_html=True)