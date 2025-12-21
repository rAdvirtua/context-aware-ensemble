import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
import os
import json
import pickle
import warnings
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import time

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

MODEL_FILE = "hybrid_model.pth"
SCALER_FILE = "scaler.pkl"
CONFIG_FILE = "model_config.json"
SEQ_LEN = 30
COOLDOWN_SECONDS = 60 # For Analysis
RETRAIN_COOLDOWN_SECONDS = 3600 # 1 Hour for Retraining (Heavy Task)

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

def get_live_news_sentiment(ticker="SPY"):
    url = f"https://news.google.com/rss/search?q={ticker}+stock+news&hl=en-US&gl=US&ceid=US:en"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://www.google.com/'
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return 0.0, [f"Blocked by provider (Status: {response.status_code})"]

        soup = BeautifulSoup(response.content, features="xml")
        items = soup.findAll('item')
        
        if not items: return 0.0, ["No news found."]
        
        parsed_news = []
        for item in items[:15]:
            title = item.title.text
            clean_title = title.split(" - ")[0]
            parsed_news.append(clean_title)
            
        vader = SentimentIntensityAnalyzer()
        scores = [vader.polarity_scores(h)['compound'] for h in parsed_news]
        
        if not scores: return 0.0, ["No scorable news found."]
        
        avg_score = np.mean(scores)
        return avg_score, parsed_news[:5]
        
    except Exception as e:
        return 0.0, [f"Error fetching news: {str(e)}"]

@st.cache_resource
def load_resources():
    if not os.path.exists(MODEL_FILE): return None, None, None
    with open(CONFIG_FILE, 'r') as f: config = json.load(f)
    model = ContextAwareEnsemble(num_inputs=config["input_dim"], num_channels_tcn=[32,64,32], d_model_trans=32).to(device)
    model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
    model.eval()
    with open(SCALER_FILE, 'rb') as f: scaler = pickle.load(f)
    return model, scaler, config

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

def generate_reasoning(pred_return, vix, z_score, momentum):
    reasoning = ""
    
    if vix < 20: regime = "Calm/Bullish"
    elif vix < 28: regime = "Volatile/Caution"
    else: regime = "Panic/Bearish"
    
    trend = "Positive" if momentum > 0 else "Negative"
    
    if z_score > 1.0: context = "Euphoric News"
    elif z_score < -1.5: context = "Fearful News"
    else: context = "Neutral News"

    if pred_return > 0:
        if regime == "Calm/Bullish" and trend == "Positive":
            reasoning = f"**Confluence:** The market is calm (VIX {vix:.0f}) and price momentum is positive. The AI sees a clear path for growth."
        elif regime == "Panic/Bearish" and z_score > -1.0:
            reasoning = f"**Contrarian Rebound:** Although volatility is high (VIX {vix:.0f}), sentiment is stabilizing. The AI detects an oversold bounce opportunity."
        else:
            reasoning = f"**Technical Strength:** Despite {context.lower()}, strong momentum metrics suggest the uptrend will continue."
    else:
        if regime == "Panic/Bearish":
            reasoning = f"**Risk Off:** Extreme volatility (VIX {vix:.0f}) combined with {context.lower()} signals a potential crash or drawdown."
        elif trend == "Negative":
            reasoning = f"**Weak Structure:** Even if news is okay, price momentum is fading. The AI predicts 'Dead Money' or a slow bleed."
        else:
            reasoning = "**Conflicting Signals:** Volatility and Sentiment are mismatched. The AI is defaulting to a defensive posture to preserve capital."
            
    return reasoning

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
    
    df['Sentiment_Score'] = calculate_implied_sentiment(df)
    df.iloc[-1, df.columns.get_loc('Sentiment_Score')] = live_sentiment
    
    df['Signal_Smooth'] = df['Sentiment_Score'].ewm(span=14, adjust=False).mean()
    df['Rolling_Mean'] = df['Signal_Smooth'].rolling(window=365).mean()
    df['Rolling_Std'] = df['Signal_Smooth'].rolling(window=365).std()
    
    df['Rolling_Mean'] = df['Rolling_Mean'].fillna(method='bfill')
    df['Rolling_Std'] = df['Rolling_Std'].fillna(method='bfill')
    
    df['Sent_Z_Score'] = (df['Signal_Smooth'] - df['Rolling_Mean']) / df['Rolling_Std']
    df['Sent_Z_Score'] = df['Sent_Z_Score'].fillna(0)
    df['Panic_Signal'] = np.where(df['Sent_Z_Score'] < -1.5, -1, 1)
    
    return df.dropna(), headlines, live_sentiment

st.set_page_config(page_title="AI Market Predictor", layout="centered")

if 'last_run' not in st.session_state: st.session_state['last_run'] = 0
if 'last_retrain' not in st.session_state: st.session_state['last_retrain'] = 0

tab1, tab2 = st.tabs(["🚀 AI Prediction Dashboard", "📚 Beginner's Guide"])

with tab1:
    st.title("🤖 Live AI Market Trader")
    st.markdown("This tool tells you if it's safe to invest in the **S&P 500 (`^GSPC`)** today.")

    st.sidebar.header("⚙️ Settings")
    sentiment_mode = st.sidebar.radio("Sentiment Source:", ["Auto-Scrape News", "Manual Input Slider"])
    
    manual_val = 0.0
    if sentiment_mode == "Manual Input Slider":
        manual_val = st.sidebar.slider("How is the Market Vibe?", -1.0, 1.0, 0.0, 0.1)
        st.sidebar.info(f"Using Manual Score: {manual_val}")
    else:
        st.sidebar.info("🕷️ Will scrape Google News for 'SPY'.")

    st.sidebar.markdown("---")
    st.sidebar.header("🔧 Active Learning")
    
    if st.sidebar.button("Retrain AI Brain"):
        curr_ts = time.time()
        # --- SPAM PROTECTION CHECK ---
        time_since_retrain = curr_ts - st.session_state['last_retrain']
        if time_since_retrain < RETRAIN_COOLDOWN_SECONDS:
            mins_left = int((RETRAIN_COOLDOWN_SECONDS - time_since_retrain) / 60)
            st.sidebar.error(f"⚠️ Cooldown Active! Wait {mins_left} min.")
        else:
            st.session_state['last_retrain'] = curr_ts
            model, scaler, config = load_resources()
            if model is None:
                st.error("Model missing. Cannot retrain.")
            else:
                with st.spinner("🧠 Fine-tuning on recent data..."):
                    try:
                        df, _, _ = run_analysis("Auto-Scrape News")
                        batch_data = df.iloc[-60:][['Returns', 'VIX', 'Mom_10', 'Sent_Z_Score', 'Panic_Signal']]
                        scaled_batch = scaler.transform(batch_data)
                        
                        xs, ys = [], []
                        for i in range(len(scaled_batch) - SEQ_LEN):
                            xs.append(scaled_batch[i:i+SEQ_LEN])
                            ys.append(scaled_batch[i+SEQ_LEN, 0])
                            
                        X_train = torch.FloatTensor(np.array(xs)).to(device)
                        y_train = torch.FloatTensor(np.array(ys)).unsqueeze(1).to(device)
                        
                        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
                        criterion = nn.MSELoss()
                        
                        model.train()
                        for _ in range(5):
                            optimizer.zero_grad()
                            loss = criterion(model(X_train), y_train)
                            loss.backward()
                            optimizer.step()
                            
                        torch.save(model.state_dict(), MODEL_FILE)
                        st.sidebar.success(f"✅ Brain Updated! Loss: {loss.item():.4f}")
                    except Exception as e:
                        st.sidebar.error(f"Retraining Failed: {e}")

    if st.button("🚀 Analyze Market Now"):
        current_time = time.time()
        time_since_last = current_time - st.session_state['last_run']
        
        if time_since_last < COOLDOWN_SECONDS:
            wait_time = int(COOLDOWN_SECONDS - time_since_last)
            st.error(f"⚠️ **Please Wait:** Anti-Spam Cooldown active ({wait_time}s remaining).")
        else:
            st.session_state['last_run'] = current_time
            
            model, scaler, config = load_resources()
            if model is None:
                st.error("⚠️ Model files missing! Please upload hybrid_model.pth.")
            else:
                with st.spinner("🕷️ Reading News & Analyzing Charts..."):
                    df, headlines, final_sent_score = run_analysis(sentiment_mode, manual_val)
                    
                    input_slice = df.iloc[-SEQ_LEN:][['Returns', 'VIX', 'Mom_10', 'Sent_Z_Score', 'Panic_Signal']]
                    input_scaled = scaler.transform(input_slice)
                    input_tensor = torch.FloatTensor(input_scaled).unsqueeze(0).to(device)
                    
                    model.eval()
                    with torch.no_grad(): pred_raw = model(input_tensor).item()
                    
                    mean, std = scaler.mean_[0], scaler.scale_[0]
                    pred_real = (pred_raw * std) + mean
                    
                    curr_vix = df['VIX'].iloc[-1]
                    curr_z = df['Sent_Z_Score'].iloc[-1]
                    curr_mom = df['Mom_10'].iloc[-1]

                    st.success("✅ Analysis Complete!")

                    st.markdown("---")
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        if pred_real > 0:
                            st.markdown("# 🟢 BUY")
                            st.caption(f"Bullish Forecast (+{pred_real*100:.2f}%)")
                        else:
                            st.markdown("# 🔴 WAIT")
                            st.caption(f"Bearish Forecast ({pred_real*100:.2f}%)")
                            
                    with col2:
                        st.markdown("### **AI Logic:**")
                        reasoning = generate_reasoning(pred_real, curr_vix, curr_z, curr_mom)
                        st.info(reasoning)

                    st.markdown("---")
                    st.subheader("📊 The Data Behind the Decision")
                    
                    m1, m2, m3 = st.columns(3)
                    
                    m1.metric("VIX (Fear Level)", f"{curr_vix:.2f}")
                    with m1.expander("What is VIX?"):
                        st.write("The 'Fear Gauge'. Below 20 is calm; Above 30 is panic.")

                    m2.metric("Sentiment Z-Score", f"{curr_z:.2f}")
                    with m2.expander("What is Z-Score?"):
                        st.write("Context of news. Derived from Implied Sentiment (VIX/RSI) to reconstruct historical context.")
                        
                    m3.metric("AI Confidence", f"{final_sent_score:.2f}")
                    with m3.expander("What is this?"):
                        st.write("The raw score (-1 to 1) from reading the headlines below.")

                    if sentiment_mode == "Auto-Scrape News":
                        st.markdown("### 📰 Headlines AI Read Today")
                        for h in headlines:
                            st.text(f"• {h}")

with tab2:
    st.title("📚 How This Works (For Beginners)")
    
    st.markdown("""
    ### 1. What is this app?
    This app uses Artificial Intelligence to answer one simple question: 
    **"Is it safe to put money in the stock market today?"**
    
    It doesn't guess randomly. It combines two superpowers:
    1.  **Reading:** It reads live news headlines (Bloomberg, Reuters) to understand the "Vibe."
    2.  **Math:** It looks at charts and Volatility (Fear) to check the facts.
    
    ### 2. The Decision Logic
    The AI gives you one of two signals based on the conflict between Risk (VIX) and News (Sentiment).
    
    #### 🟢 BUY (Green Light)
    * **The Setup:** The AI sees a path for profit. This usually means Low Fear (VIX) + Positive Momentum.
    * **Action:** Consider entering a position in **SPY** or **VOO**.
    
    #### 🔴 WAIT / SELL (Red Light)
    * **The Setup:** The AI predicts a drop or "Dead Money" (Sideways movement).
    * **Why?** Even if news is good, if the market is too fearful (High VIX), stocks won't go up.
    * **Action:** Stay in **Cash** (Risk-Off). Protect your capital.
    
    ### 3. The "Weather" Analogy 
    Imagine the Stock Market is an Ocean, and you are a Sailor.
    * **VIX (Fear):** This is the **Storm Forecast**. If VIX is high, there is a hurricane. Don't sail!
    * **Sentiment:** This is the **Wind**. If news is positive, the wind is at your back (Good).
    * **Decision:** Even if the wind is good, we don't sail into a hurricane.

    ### 4. The Billion-Dollar Problem (and our free fix)
    Big hedge funds pay millions of dollars for "Bloomberg Terminals" to download 20 years of news archives. 
    
    **We are limited by Paywalls.** We cannot scrape news from 6 months ago for free.
    
    **Our Solution: "Implied Sentiment"**
    Instead of paying for old news, we look at **Market Behavior**.
    * If the market was crashing in the past, we assume the news was *Bad*.
    * If the market was soaring, we assume the news was *Good*.
    
    We use this logic to reconstruct a "Simulated History" of sentiment. This allows our AI to have a mathematical baseline (Z-Score) without needing a corporate budget!
    """)

st.markdown("---")
st.markdown("""
### ⚖️ Disclaimer: Not Financial Advice
**The content provided by this application is for educational and informational purposes only.** It does not constitute financial, investment, or trading advice. The 'AI' signals displayed here are generated based on historical data and probabilistic models, which can and do fail. 

* **You are solely responsible for your own investment decisions.**
* The creator of this project accepts **no liability** for any financial losses or damages incurred resulting from the use of this tool.
* Always consult with a qualified financial advisor before making investment decisions.
* Trading stocks and derivatives involves **high risk** and you can lose your entire investment.
""")
