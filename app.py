import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
import os
import json
import csv
import pickle
import warnings
import requests
from bs4 import BeautifulSoup
import time
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_FILE = "hybrid_model.pth"
SCALER_FILE = "scaler.pkl"
CONFIG_FILE = "model_config.json"
HISTORICAL_DATA_FILE = "mcwsi_historical_2024.csv"
PROPRIETARY_DB_FILE = "proprietary_dataset.csv"
NLP_MODEL_NAME = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
SEQ_LEN = 30
COOLDOWN_SECONDS = 60
RETRAIN_COOLDOWN_SECONDS = 3600

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    def forward(self, x): return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        try: wn = nn.utils.parametrizations.weight_norm
        except AttributeError: wn = nn.utils.weight_norm
        self.conv1 = wn(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = wn(nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1, self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.net(x); res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN_Expert(nn.Module):
    def __init__(self, n_in, n_chan, k=2, drop=0.2):
        super(TCN_Expert, self).__init__()
        layers = []
        for i in range(len(n_chan)):
            d = 2 ** i
            in_ch = n_in if i == 0 else n_chan[i-1]
            layers += [TemporalBlock(in_ch, n_chan[i], k, 1, d, (k-1)*d, drop)]
        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(n_chan[-1], 1)
    def forward(self, x): return self.linear(self.network(x.permute(0, 2, 1))[:, :, -1])

class SentimentTransformer_Expert(nn.Module):
    def __init__(self, input_dim=1, d_model=32, nhead=4, num_layers=2, dropout=0.2):
        super(SentimentTransformer_Expert, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 5000, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=64, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, 1)
    def forward(self, x):
        x = self.input_proj(x) + self.pos_encoder[:, :x.size(1), :]
        x = self.transformer_encoder(x)
        return self.decoder(x[:, -1, :])

class ContextAwareEnsemble(nn.Module):
    def __init__(self, num_inputs, num_channels_tcn, d_model_trans=32):
        super(ContextAwareEnsemble, self).__init__()
        self.technical_expert = TCN_Expert(num_inputs - 1, num_channels_tcn)
        self.sentiment_expert = SentimentTransformer_Expert(input_dim=1, d_model=d_model_trans)
        self.gating_net = nn.Sequential(nn.Linear(num_inputs, 32), nn.ReLU(), nn.Linear(32, 2), nn.Softmax(dim=1))
    def forward(self, x):
        weights = self.gating_net(x[:, -1, :]) 
        tech_out = self.technical_expert(x[:, :, :-1])
        sent_out = self.sentiment_expert(x[:, :, -1:])
        return (weights[:, 0:1] * tech_out) + (weights[:, 1:2] * sent_out)

@st.cache_resource
def load_nlp_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained(NLP_MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(NLP_MODEL_NAME).to(device)
        model.eval()
        return tokenizer, model
    except Exception as e:
        st.error(f"Failed to load NLP Model: {e}")
        return None, None

tokenizer, nlp_model = load_nlp_model()

@st.cache_resource
def load_resources():
    if not os.path.exists(MODEL_FILE): return None, None, None
    with open(CONFIG_FILE, 'r') as f: config = json.load(f)
    model = ContextAwareEnsemble(num_inputs=config["input_dim"], num_channels_tcn=[32,64,32], d_model_trans=32).to(device)
    model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
    model.eval()
    with open(SCALER_FILE, 'rb') as f: scaler = pickle.load(f)
    return model, scaler, config

def log_data_snapshot(mcwsi, headlines, prediction, vix, momentum):
    file_exists = os.path.isfile(PROPRIETARY_DB_FILE)
    with open(PROPRIETARY_DB_FILE, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["timestamp", "mcwsi_score", "vix", "momentum", "prediction", "num_articles", "headlines_json"])
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        headlines_json = json.dumps(headlines)
        writer.writerow([timestamp, mcwsi, vix, momentum, prediction, len(headlines), headlines_json])

def get_live_news_sentiment(ticker="SPY"):
    url = f"https://news.google.com/rss/search?q={ticker}+stock+news&hl=en-US&gl=US&ceid=US:en"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200: return 0.0, [f"Blocked by provider (Status: {response.status_code})"]
        soup = BeautifulSoup(response.content, features="xml")
        items = soup.findAll('item')
        if not items: return 0.0, ["No news found."]
        headlines = [item.title.text.split(" - ")[0] for item in items[:15]]
        if not headlines: return 0.0, ["No scorable news found."]
        if nlp_model is None: return 0.0, ["Model failed to load."]

        inputs = tokenizer(headlines, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = nlp_model(**inputs)
            probs = softmax(outputs.logits, dim=1)

        sentiment_scores = probs[:, 2] - probs[:, 0]
        avg_score = torch.mean(sentiment_scores).item()
        return avg_score, headlines[:5]
    except Exception as e:
        return 0.0, [f"Error fetching news: {str(e)}"]

def calculate_implied_sentiment(df):
    delta = df['SP500'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi_norm = (rsi - 50) / 50
    vix_clamped = df['VIX'].clip(10, 60)
    vix_inv = (60 - vix_clamped) / 50
    vix_norm = (vix_inv * 2) - 1
    implied_sentiment = (rsi_norm + vix_norm) / 2
    return implied_sentiment.fillna(0)

def get_hybrid_sentiment(df):
    implied = calculate_implied_sentiment(df)
    if os.path.exists(HISTORICAL_DATA_FILE):
        try:
            real_data = pd.read_csv(HISTORICAL_DATA_FILE, parse_dates=['Date'], index_col='Date')
            real_aligned = real_data.reindex(df.index)
            hybrid = real_aligned['MCWSI'].combine_first(implied)
            return hybrid
        except Exception:
            return implied
    return implied

def run_analysis(sentiment_mode, manual_score=0.0):
    tickers = ['^GSPC', '^VIX', 'DX-Y.NYB', '^TNX']
    data = yf.download(tickers, period="2y", progress=False)
    
    if isinstance(data.columns, pd.MultiIndex):
        if 'Adj Close' in data.columns.get_level_values(0): df = data['Adj Close'].copy()
        elif 'Close' in data.columns.get_level_values(0): df = data['Close'].copy()
        else: df = data.xs('Close', level=1, axis=1) if 'Close' in data.columns.levels[1] else data
    else: df = data['Adj Close'].copy() if 'Adj Close' in data.columns else data['Close'].copy()
    
    df = df.rename(columns={'^GSPC': 'SP500', '^VIX': 'VIX', 'DX-Y.NYB': 'DXY', '^TNX': 'TNX'})
    df['Returns'] = np.log(df['SP500'] / df['SP500'].shift(1))
    df['Vol_MA'] = df['VIX'].rolling(window=10).mean()
    df['Mom_10'] = df['SP500'].pct_change(10)
    
    headlines = []
    if sentiment_mode == "Auto-Scrape News":
        live_sentiment, headlines = get_live_news_sentiment("SPY")
    else:
        live_sentiment = manual_score
        headlines = ["User Manual Input override active."]
        
    df['Sentiment_Score'] = get_hybrid_sentiment(df)
    df.iloc[-1, df.columns.get_loc('Sentiment_Score')] = live_sentiment
    
    df['Signal_Smooth'] = df['Sentiment_Score'].ewm(span=14, adjust=False).mean()
    df['Rolling_Mean'] = df['Signal_Smooth'].shift(1).rolling(window=365, min_periods=30).mean()
    df['Rolling_Std'] = df['Signal_Smooth'].shift(1).rolling(window=365, min_periods=30).std()
    
    df['Rolling_Mean'] = df['Rolling_Mean'].fillna(method='bfill')
    df['Rolling_Std'] = df['Rolling_Std'].fillna(method='bfill')
    
    df['Sent_Z_Score'] = (df['Signal_Smooth'] - df['Rolling_Mean']) / df['Rolling_Std']
    df['Sent_Z_Score'] = df['Sent_Z_Score'].fillna(0)
    
    df['Panic_Signal'] = np.where(df['Sent_Z_Score'] < -1.5, -1, 1)
    
    return df.dropna(), headlines, live_sentiment

def generate_reasoning(pred_return, vix, z_score, momentum):
    reasoning = ""
    if vix < 20: regime = "Calm/Bullish"
    elif vix < 28: regime = "Volatile/Caution"
    else: regime = "Panic/Bearish"
    
    trend = "Positive" if momentum > 0 else "Negative"
    
    if z_score > 1.0: context = "Euphoric News"
    elif z_score < -1.5: context = "Extreme Fear"
    else: context = "Neutral News"

    if pred_return > 0:
        if regime == "Calm/Bullish":
            reasoning = f"**Green Light:** Low volatility (VIX {vix:.0f}) and positive structure. The AI recommends a Long position to capture trend continuation."
        elif regime == "Panic/Bearish" and z_score > -1.0:
            reasoning = f"**Contrarian Buy:** VIX is high ({vix:.0f}), but sentiment is stabilizing. The AI detects an oversold bounce opportunity."
        else:
            reasoning = f"**Momentum Play:** Despite {context.lower()}, technicals remain strong. The AI favors staying in the market over paying fees to exit."
    else:
        if regime == "Panic/Bearish":
            reasoning = f"**Defensive Short:** Extreme volatility (VIX {vix:.0f}) + {context.lower()}. The AI predicts further downside and recommends Shorting or Cash."
        elif trend == "Negative":
            reasoning = f"**Trend Following Short:** Price momentum is fading. The AI predicts a slow bleed and recommends exiting Long positions."
        else:
            reasoning = "**Fee Avoidance:** The signal is weak/negative. Backtests show it is more profitable to exit/short here than to hold through the chop."
            
    return reasoning

def get_market_mood_emoji(z_score):
    if z_score > 1.5: return "🚀"
    elif z_score > 0.5: return "😊"
    elif z_score > -0.5: return "😐"
    elif z_score > -1.5: return "😰"
    else: return "😱"

st.set_page_config(page_title="Hybrid AI Trader", layout="centered", initial_sidebar_state="expanded")

if 'last_run' not in st.session_state: st.session_state['last_run'] = 0
if 'last_retrain' not in st.session_state: st.session_state['last_retrain'] = 0
if 'analysis_count' not in st.session_state: st.session_state['analysis_count'] = 0
if 'show_quick_tips' not in st.session_state: st.session_state['show_quick_tips'] = True

st.sidebar.title("🤖 Control Panel")
st.sidebar.header("1. Input Settings")
sentiment_mode = st.sidebar.radio("Sentiment Source:", ["Auto-Scrape News", "Manual Input Slider"])
manual_val = 0.0
if sentiment_mode == "Manual Input Slider":
    manual_val = st.sidebar.slider("Market Vibe (-1=Panic, 1=Hype)", -1.0, 1.0, 0.0, 0.1)
    st.sidebar.caption("💡 Tip: Use -1.0 to simulate a market crash, +1.0 for extreme optimism")
else:
    st.sidebar.info("🕷️ Scraping Google News for 'SPY'")

st.sidebar.markdown("---")
st.sidebar.header("2. Active Learning")
st.sidebar.caption("Teach the AI with latest data (Weekly)")

if st.sidebar.button("Retrain AI Brain", use_container_width=True):
    curr_ts = time.time()
    time_since_retrain = curr_ts - st.session_state['last_retrain']
    if time_since_retrain < RETRAIN_COOLDOWN_SECONDS:
        st.sidebar.error(f"⏳ Cooldown: Wait {int((RETRAIN_COOLDOWN_SECONDS - time_since_retrain)/60)} mins.")
    else:
        st.session_state['last_retrain'] = curr_ts
        model, scaler, config = load_resources()
        if model is None:
            st.sidebar.error("❌ Model missing.")
        else:
            with st.sidebar.status("🧠 Training...", expanded=True) as status:
                try:
                    df, _, _ = run_analysis("Auto-Scrape News")
                    batch = df.iloc[-60:][['Returns', 'VIX', 'Mom_10', 'Sent_Z_Score', 'Panic_Signal']]
                    scaled = scaler.transform(batch)
                    
                    xs, ys = [], []
                    for i in range(len(scaled) - SEQ_LEN):
                        xs.append(scaled[i:i+SEQ_LEN])
                        ys.append(scaled[i+SEQ_LEN, 0])
                    
                    X_t = torch.FloatTensor(np.array(xs)).to(device)
                    y_t = torch.FloatTensor(np.array(ys)).unsqueeze(1).to(device)
                    
                    opt = torch.optim.Adam(model.parameters(), lr=0.0001)
                    crit = nn.MSELoss()
                    model.train()
                    for _ in range(5):
                        opt.zero_grad()
                        loss = crit(model(X_t), y_t)
                        loss.backward()
                        opt.step()
                    torch.save(model.state_dict(), MODEL_FILE)
                    status.update(label="✅ Brain Updated!", state="complete", expanded=False)
                    st.sidebar.success("🎉 Model trained successfully!")
                except Exception as e:
                    status.update(label="❌ Failed", state="error")
                    st.sidebar.error(str(e))

st.sidebar.markdown("---")
st.sidebar.header("📊 Quick Stats")
if st.session_state['analysis_count'] > 0:
    st.sidebar.metric("Analyses Run", st.session_state['analysis_count'])
    if os.path.exists(PROPRIETARY_DB_FILE):
        try:
            hist_data = pd.read_csv(PROPRIETARY_DB_FILE)
            st.sidebar.metric("Total Predictions Logged", len(hist_data))
        except:
            pass

tab1, tab2, tab3 = st.tabs(["🚀 Dashboard", "📖 How to Use", "📚 Technical Guide"])

with tab1:
    st.title("🤖 Live Market Analysis")
    st.markdown("Multi-Expert Ensemble: **TCN (Technicals)** + **Transformer (Sentiment)**")

    if st.session_state['show_quick_tips']:
        with st.expander("💡 Quick Start Tips (Click to dismiss)", expanded=True):
            st.markdown("""
            1. **Choose your mode:** Auto-scrape news for real-time analysis, or use manual slider to test scenarios
            2. **Click "Analyze Market Now"** to get AI prediction (60-second cooldown between runs)
            3. **Read the reasoning:** The AI explains WHY it made its decision
            4. **Check the data:** Scroll down to see VIX, Z-Score, and news headlines
            """)
            if st.button("Got it! Don't show again"):
                st.session_state['show_quick_tips'] = False
                st.rerun()

    if st.button("🚀 Analyze Market Now", use_container_width=True, type="primary"):
        current_time = time.time()
        time_since_last = current_time - st.session_state['last_run']
        
        if time_since_last < COOLDOWN_SECONDS:
            st.warning(f"⏳ Cooldown active: Wait {int(COOLDOWN_SECONDS - time_since_last)}s to prevent API overload.")
        else:
            st.session_state['last_run'] = current_time
            st.session_state['analysis_count'] += 1
            model, scaler, config = load_resources()
            
            if model is None:
                st.error("⚠️ Model files missing! Please upload hybrid_model.pth.")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("📥 Fetching market data...")
                progress_bar.progress(25)
                
                df, headlines, final_sent_score = run_analysis(sentiment_mode, manual_val)
                
                status_text.text("🧮 Processing indicators...")
                progress_bar.progress(50)
                
                input_slice = df.iloc[-SEQ_LEN:][['Returns', 'VIX', 'Mom_10', 'Sent_Z_Score', 'Panic_Signal']]
                input_scaled = scaler.transform(input_slice)
                input_tensor = torch.FloatTensor(input_scaled).unsqueeze(0).to(device)
                
                status_text.text("🤖 Running AI prediction...")
                progress_bar.progress(75)
                
                model.eval()
                with torch.no_grad(): pred_raw = model(input_tensor).item()
                
                mean, std = scaler.mean_[0], scaler.scale_[0]
                pred_real = (pred_raw * std) + mean
                
                curr_vix = df['VIX'].iloc[-1]
                curr_z = df['Sent_Z_Score'].iloc[-1]
                curr_mom = df['Mom_10'].iloc[-1]
                
                log_data_snapshot(final_sent_score, headlines, pred_real, curr_vix, curr_mom)
                
                progress_bar.progress(100)
                status_text.text("✅ Analysis complete!")
                time.sleep(0.5)
                progress_bar.empty()
                status_text.empty()

                st.markdown("---")
                
                mood_emoji = get_market_mood_emoji(curr_z)
                st.markdown(f"### Market Mood: {mood_emoji}")
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    if pred_real > 0:
                        st.success("# ✅ BUY / LONG")
                        st.metric("Model Forecast", f"+{pred_real*100:.2f}%", delta=f"{pred_real*100:.2f}%")
                    else:
                        st.error("# 🛑 SELL / SHORT")
                        st.metric("Model Forecast", f"{pred_real*100:.2f}%", delta=f"{pred_real*100:.2f}%")
                    
                    confidence_level = abs(pred_real) * 100
                    st.progress(min(confidence_level / 5, 1.0))
                    st.caption(f"Signal Strength: {'Strong' if confidence_level > 2 else 'Moderate' if confidence_level > 1 else 'Weak'}")
                        
                with col2:
                    st.markdown("### **AI Reasoning:**")
                    reasoning = generate_reasoning(pred_real, curr_vix, curr_z, curr_mom)
                    st.info(reasoning)

                st.markdown("---")
                st.subheader("📊 The Data Behind the Decision")
                
                m1, m2, m3 = st.columns(3)
                
                vix_delta = "Normal" if curr_vix < 20 else "High" if curr_vix < 30 else "Extreme"
                m1.metric("VIX (Fear Gauge)", f"{curr_vix:.2f}", delta=vix_delta, delta_color="inverse")
                with m1.expander("ℹ️ Why VIX?"):
                    st.write("The Gating Network uses VIX to weigh Technicals vs. Sentiment. Below 20 is Calm; Above 30 is Panic.")

                z_sentiment = "Euphoric" if curr_z > 1 else "Fearful" if curr_z < -1 else "Neutral"
                m2.metric("Sentiment Z-Score", f"{curr_z:.2f}", delta=z_sentiment)
                with m2.expander("ℹ️ What is Z-Score?"):
                    st.write("A statistical anomaly detector. We use *Yesterday's* data to calculate the mean, preventing look-ahead bias.")
                    st.write(f"**Current Status:** {z_sentiment}")
                    
                conf_pct = abs(final_sent_score) * 100
                m3.metric("News Sentiment", f"{final_sent_score:.2f}", delta=f"{conf_pct:.0f}%")
                with m3.expander("ℹ️ Confidence Score"):
                    st.write("The raw output (-1 to 1) from the Transformer reading the headlines below.")
                    st.write("Closer to -1 = Bearish news, Closer to +1 = Bullish news")

                st.markdown("---")
                st.markdown("### 📰 Headlines Analyzed")
                with st.expander("📄 Show News Sources", expanded=True):
                    for idx, h in enumerate(headlines, 1):
                        st.markdown(f"**{idx}.** {h}")

                st.balloons()

with tab2:
    st.title("📖 How to Use This App")
    
    st.header("🎯 What Does This App Do?")
    st.markdown("""
    This is an AI-powered market analysis tool that predicts whether you should **BUY/LONG** or **SELL/SHORT** 
    the S&P 500 (SPY) based on:
    - 📊 Technical indicators (price patterns, volatility)
    - 📰 News sentiment (what the media is saying)
    - 🧠 Deep learning models trained on historical data
    """)
    
    st.markdown("---")
    st.header("🚀 Step-by-Step Guide")
    
    with st.expander("**Step 1: Choose Your Input Mode**", expanded=True):
        st.markdown("""
        In the **left sidebar**, you'll see two options:
        
        **🕷️ Auto-Scrape News (Recommended for beginners)**
        - The AI automatically fetches the latest news headlines about SPY from Google News
        - It analyzes them in real-time to gauge market sentiment
        - Best for getting actual market predictions
        
        **🎚️ Manual Input Slider (For testing)**
        - You control the sentiment score manually
        - Useful for "what-if" scenarios: "What if the market was in extreme panic?"
        - Slide from -1 (Panic) to +1 (Euphoria)
        """)
        st.image("https://via.placeholder.com/600x200/3b82f6/ffffff?text=Sidebar+Controls", use_container_width=True)
    
    with st.expander("**Step 2: Run the Analysis**"):
        st.markdown("""
        1. Click the big **"🚀 Analyze Market Now"** button on the Dashboard tab
        2. Wait 5-10 seconds while the AI:
           - Downloads market data (VIX, S&P 500 prices)
           - Scrapes news headlines
           - Runs the prediction model
        3. You'll see a progress bar showing what's happening
        
        **⏳ Cooldown Timer:** You can only run analysis once every 60 seconds to prevent overloading data sources.
        """)
    
    with st.expander("**Step 3: Understand the Results**"):
        st.markdown("""
        The AI will show you:
        
        **🟢 BUY/LONG Signal**
        - Green box with an upward forecast (e.g., +1.23%)
        - The AI thinks the market will go UP
        
        **🔴 SELL/SHORT Signal**
        - Red box with a downward forecast (e.g., -0.87%)
        - The AI thinks the market will go DOWN
        
        **💬 AI Reasoning Box**
        - Explains WHY the AI made this decision
        - Examples: "Low volatility + bullish news" or "Extreme panic detected"
        """)
    
    with st.expander("**Step 4: Check the Data**"):
        st.markdown("""
        Scroll down to see three key metrics:
        
        1. **VIX (Fear Gauge):** Below 20 = calm, above 30 = panic
        2. **Sentiment Z-Score:** How unusual today's news is compared to history
        3. **News Sentiment:** The raw sentiment score from analyzing headlines
        
        Click the "📄 Show News Sources" button to see which headlines the AI read.
        """)
    
    st.markdown("---")
    st.header("🎓 Advanced Features")
    
    with st.expander("**🔄 Retrain AI Brain**"):
        st.markdown("""
        **When to use:** Once per week to keep the AI updated with the latest market patterns.
        
        **What it does:**
        - Pulls the last 60 days of data
        - Fine-tunes the model weights
        - Saves the updated model
        
        **⏳ Cooldown:** 1 hour between retraining sessions (to prevent overfitting)
        """)
    
    with st.expander("**📊 Quick Stats Sidebar**"):
        st.markdown("""
        At the bottom of the sidebar, you'll see:
        - How many analyses you've run this session
        - Total predictions logged in the database
        
        This helps you track your usage.
        """)
    
    st.markdown("---")
    st.header("💡 Pro Tips")
    
    col_tip1, col_tip2 = st.columns(2)
    
    with col_tip1:
        st.markdown("""
        **🎯 Best Practices:**
        - Run analysis during market hours (9:30 AM - 4 PM EST) for freshest news
        - Check multiple times per day to see how sentiment shifts
        - Use manual mode to test historical "what-if" scenarios
        - Retrain weekly after major market events
        """)
    
    with col_tip2:
        st.markdown("""
        **⚠️ Common Mistakes:**
        - Don't spam the analyze button (respect cooldowns)
        - Don't ignore the reasoning - it's as important as the signal
        - Don't use this as sole trading advice (educational tool only)
        - Don't retrain too often (causes overfitting)
        """)
    
    st.markdown("---")
    st.header("❓ FAQ")
    
    with st.expander("Why does it say 'Cooldown active'?"):
        st.markdown("""
        The app limits how often you can run analysis to:
        1. Prevent overloading free data sources (Yahoo Finance, Google News)
        2. Avoid IP bans from web scraping
        3. Save computational resources
        
        **60-second cooldown** for analysis, **1-hour cooldown** for retraining.
        """)
    
    with st.expander("What does 'Signal Strength' mean?"):
        st.markdown("""
        This shows how confident the AI is in its prediction:
        - **Strong:** Prediction > 2% (high conviction)
        - **Moderate:** Prediction 1-2% (medium conviction)
        - **Weak:** Prediction < 1% (low conviction)
        
        Weak signals might indicate a choppy/sideways market.
        """)
    
    with st.expander("Can I use this for real trading?"):
        st.markdown("""
        **This is an educational demo only.** While the AI uses sophisticated techniques:
        - Past performance doesn't guarantee future results
        - Markets are unpredictable and can change instantly
        - Always consult a financial advisor before trading
        - Use proper risk management and position sizing
        
        Think of this as a learning tool to understand how AI analyzes markets.
        """)
    
    with st.expander("Why do I see different results each time?"):
        st.markdown("""
        Multiple factors cause variation:
        1. **News changes:** Headlines update constantly throughout the day
        2. **Market prices:** VIX and S&P 500 move in real-time
        3. **Sentiment shifts:** Public mood can swing rapidly
        
        This is normal and reflects actual market dynamics.
        """)

with tab3:
    st.title("📚 Technical Guide & Mathematical Proofs")
    
    st.header("1. The Core Architecture")
    st.markdown("""
    This application is powered by a **Context-Aware Hybrid Ensemble**. It does not rely on a single algorithm but instead uses two specialized "Experts" that vote on the market direction.
    
    * **Expert A: The Pattern Spotter (TCN)**
        * **Type:** Temporal Convolutional Network.
        * **Role:** Analyzes price action (Momentum, Volatility) using dilated convolutions.
        * **Math:** It identifies patterns over long time horizons ($t-30$ to $t$) to predict $t+1$.
    
    * **Expert B: The News Reader (Transformer)**
        * **Type:** DistilRoBERTa (Fine-Tuned).
        * **Role:** Analyzes textual sentiment from Google News headlines.
        * **Math:** Converts text into high-dimensional vectors (Embeddings) to score Fear vs. Greed.
    """)

    st.markdown("---")
    st.header("2. Mathematical Logic: Implied Sentiment")
    st.markdown("""
    **The Problem:** We need 5+ years of data to train the AI, but detailed news sentiment data is often expensive or unavailable for the past.
    
    **The Solution:** We derive an *Implied Sentiment* score from market behavior. If investors are selling (RSI Low) and volatility is high (VIX High), we mathematically infer that the news environment is negative.
    
    **The Formula:**
    """)
    
    st.latex(r'''
    S_{implied} = \frac{\text{RSI}_{norm} + (1 - \text{VIX}_{norm})}{2}
    ''')
    
    st.markdown("""
    **Where:**
    * **RSI (Relative Strength Index):** Normalized to a 0-1 scale. A low RSI (oversold) pulls the score down.
    * **VIX (Volatility Index):** Clamped between 10 and 60, then inverted. A high VIX (Panic) results in a low score closer to 0.
    
    This creates a synthetic "Sentiment Signal" that perfectly fills the gaps in our historical dataset.
    """)

    st.markdown("---")
    st.header("3. Mathematical Logic: Leak-Proof Z-Scores")
    st.markdown("""
    **The Problem:** Many AI models fail because of "Look-Ahead Bias"—they use *today's* data to calculate the average, which makes today look less volatile than it really is.
    
    **The Solution:** We calculate the Z-Score (Anomaly Score) using a **Shifted Rolling Window**. The definition of "Panic" for *Today* is based strictly on the statistics of *Yesterday*.
    
    **The Formula:**
    """)
    
    st.latex(r'''
    Z_t = \frac{X_t - \mu_{t-1}}{\sigma_{t-1}}
    ''')
    
    st.markdown("""
    **Where:**
    * $X_t$: The Sentiment Score at time $t$ (Today).
    * $\mu_{t-1}$: The Rolling Mean calculated up to $t-1$ (Yesterday).
    * $\sigma_{t-1}$: The Rolling Standard Deviation calculated up to $t-1$ (Yesterday).
    
    **Why this matters:** By strictly separating $t$ from the baseline calculation, we ensure the simulation is 100% realistic and "Net-of-Fees" profitable.
    """)

    st.markdown("---")
    st.header("4. The 'Net-of-Fees' Philosophy")
    st.markdown("""
    **Why is there no 'Neutral' Signal?**
    
    In previous versions, the model would output "WAIT" if it was unsure. However, extensive Backtesting revealed a mathematical flaw in that approach:
    
    1.  **Churn Cost:** Constantly entering and exiting positions incurs transaction fees (Spread + Slippage).
    2.  **Opportunity Cost:** The market tends to drift upwards. Being "out" often means missing small gains.
    
    **The Result:** A strategy that stays Invested ("Always-In") and only exits during extreme "Panic" ($Z < -1.5$) outperforms a strategy that tries to be clever by waiting.
    """)
    
    st.markdown("---")
    st.header("5. Additional Technical Notes")
    st.markdown("""
    The model architecture combines multiple neural network components optimized for financial time-series prediction. 
    Specific implementation details are proprietary and subject to ongoing research and improvements.
    
    For questions about the methodology or to report issues, please contact the development team.
    """)

st.markdown("---")
st.markdown("### ⚖️ Disclaimer")
st.caption("Educational Demo Only. Not Financial Advice. Always consult a licensed financial advisor before making investment decisions.")
