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

# --- CONFIGURATION ---
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

# --- 1. THE BRAIN (Neural Network Architecture) ---
# Kept exact same logic as training to ensure it works
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

# --- 2. HELPER FUNCTIONS (Data & Logic) ---

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
    # This grabs news for SPY (S&P 500 ETF) so we get market-relevant news, not random noise.
    url = f"https://news.google.com/rss/search?q={ticker}+stock+news&hl=en-US&gl=US&ceid=US:en"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200: return 0.0, [f"Blocked by provider (Status: {response.status_code})"]
        
        soup = BeautifulSoup(response.content, features="xml")
        items = soup.findAll('item')
        if not items: return 0.0, ["No news found."]
        
        headlines = []
        for item in items[:15]:
            title = item.title.text.split(" - ")[0]
            headlines.append(title)
        
        if not headlines: return 0.0, ["No scorable news found."]
        if nlp_model is None: return 0.0, ["Model failed to load."]

        inputs = tokenizer(headlines, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = nlp_model(**inputs)
            probs = softmax(outputs.logits, dim=1)

        neg_score = probs[:, 0]
        pos_score = probs[:, 2]
        sentiment_scores = pos_score - neg_score
        avg_score = torch.mean(sentiment_scores).item()
        
        return avg_score, headlines[:5]
    except Exception as e:
        return 0.0, [f"Error fetching news: {str(e)}"]

def calculate_implied_sentiment(df):
    # This fills the "Gap" in history when we don't have survey data.
    # We "imply" sentiment from how the market is behaving.
    delta = df['SP500'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # Normalize RSI (0 to 1) and VIX (inverted, 0 to 1)
    rsi_norm = (rsi - 50) / 50
    vix_clamped = df['VIX'].clip(10, 60)
    vix_inv = (60 - vix_clamped) / 50
    vix_norm = (vix_inv * 2) - 1
    
    # Average them
    implied_sentiment = (rsi_norm + vix_norm) / 2
    return implied_sentiment.fillna(0)

def get_hybrid_sentiment(df):
    implied = calculate_implied_sentiment(df)
    if os.path.exists(HISTORICAL_DATA_FILE):
        try:
            real_data = pd.read_csv(HISTORICAL_DATA_FILE, parse_dates=['Date'], index_col='Date')
            real_aligned = real_data.reindex(df.index)
            # Combine: Use Real data where we have it, Implied where we don't
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
    
    # --- RIGOROUS MATH ---
    # We calculate Z-Scores using "Yesterday's" data to avoid cheating (Look-Ahead Bias)
    df['Signal_Smooth'] = df['Sentiment_Score'].ewm(span=14, adjust=False).mean()
    df['Rolling_Mean'] = df['Signal_Smooth'].shift(1).rolling(window=365, min_periods=30).mean()
    df['Rolling_Std'] = df['Signal_Smooth'].shift(1).rolling(window=365, min_periods=30).std()
    
    df['Rolling_Mean'] = df['Rolling_Mean'].fillna(method='bfill')
    df['Rolling_Std'] = df['Rolling_Std'].fillna(method='bfill')
    
    df['Sent_Z_Score'] = (df['Signal_Smooth'] - df['Rolling_Mean']) / df['Rolling_Std']
    df['Sent_Z_Score'] = df['Sent_Z_Score'].fillna(0)
    
    # If sentiment is -1.5 deviations below normal, it's a "Panic"
    df['Panic_Signal'] = np.where(df['Sent_Z_Score'] < -1.5, -1, 1)
    
    return df.dropna(), headlines, live_sentiment

def generate_simple_reasoning(pred_return, vix, z_score):
    # Translator: Math -> English
    
    # 1. Check the "Vibe" (Regime)
    if vix < 20: 
        vibe = "Calm"
    elif vix < 28: 
        vibe = "Nervous"
    else: 
        vibe = "Panicked"
    
    # 2. Check the News (Z-Score)
    if z_score < -1.5:
        news_status = "Very Bad News"
    elif z_score > 1.0:
        news_status = "Euphoric/Hyped"
    else:
        news_status = "Normal News"

    # 3. Final Verdict
    if pred_return > 0:
        if vibe == "Panicked":
            return f"**Rebound Opportunity:** Everyone else is panicked ({vibe} Market), but the AI thinks the selling is overdone. It sees a chance to buy low."
        else:
            return f"**All Systems Go:** The market is {vibe} and the news is {news_status}. The AI predicts prices will go UP."
    else:
        if vibe == "Panicked":
            return f"**Too Risky:** The market is {vibe} and news is {news_status}. The AI recommends sitting in cash until things settle down."
        else:
            return f"**Weakness Detected:** Even though the market seems {vibe}, the AI detects underlying weakness. It thinks it's safer to wait."

# --- 3. STREAMLIT UI ---
st.set_page_config(page_title="Beginner AI Trader", layout="centered")

if 'last_run' not in st.session_state: st.session_state['last_run'] = 0
if 'last_retrain' not in st.session_state: st.session_state['last_retrain'] = 0

# Tabs for separate sections
tab1, tab2, tab3 = st.tabs(["🚀 Predictor", "🧠 How It Works (The Math)", "🔧 Settings"])

# --- TAB 1: THE SIMPLE PREDICTOR ---
with tab1:
    st.title("🤖 Is the Market Safe?")
    st.markdown("Use Artificial Intelligence to check if it's a good day to buy the S&P 500.")

    if st.button("🚀 Analyze Market Now"):
        current_time = time.time()
        time_since_last = current_time - st.session_state['last_run']
        
        if time_since_last < COOLDOWN_SECONDS:
            st.warning(f"Please wait {int(COOLDOWN_SECONDS - time_since_last)} seconds before clicking again.")
        else:
            st.session_state['last_run'] = current_time
            
            model, scaler, config = load_resources()
            if model is None:
                st.error("⚠️ Model files missing! Please upload hybrid_model.pth.")
            else:
                with st.spinner("🕷️ AI is reading the news..."):
                    # Use 'Auto-Scrape' by default for the button
                    df, headlines, final_sent_score = run_analysis("Auto-Scrape News")
                    
                    # Prepare Data
                    input_slice = df.iloc[-SEQ_LEN:][['Returns', 'VIX', 'Mom_10', 'Sent_Z_Score', 'Panic_Signal']]
                    input_scaled = scaler.transform(input_slice)
                    input_tensor = torch.FloatTensor(input_scaled).unsqueeze(0).to(device)
                    
                    # Get Prediction
                    model.eval()
                    with torch.no_grad(): pred_raw = model(input_tensor).item()
                    
                    # Unscale
                    mean, std = scaler.mean_[0], scaler.scale_[0]
                    pred_real = (pred_raw * std) + mean
                    
                    curr_vix = df['VIX'].iloc[-1]
                    curr_z = df['Sent_Z_Score'].iloc[-1]
                    curr_mom = df['Mom_10'].iloc[-1]
                    
                    log_data_snapshot(final_sent_score, headlines, pred_real, curr_vix, curr_mom)

                    # --- RESULT DISPLAY ---
                    st.markdown("---")
                    
                    # Simple "Traffic Light" Logic
                    if pred_real > 0:
                        st.success("# ✅ MARKET LOOKS GOOD")
                        st.markdown(f"**The AI thinks prices will rise.** (Projected Gain: +{pred_real*100:.2f}%)")
                    else:
                        st.error("# 🛑 BETTER TO WAIT")
                        st.markdown(f"**The AI thinks prices might fall.** (Projected Loss: {pred_real*100:.2f}%)")
                    
                    st.info(generate_simple_reasoning(pred_real, curr_vix, curr_z))

                    # --- HEADLINES ---
                    with st.expander("📰 See the News the AI Read"):
                        for h in headlines:
                            st.write(f"- {h}")

# --- TAB 2: EDUCATIONAL (THE MATH & TECH) ---
with tab2:
    st.header("How the Magic Happens 🪄")
    st.markdown("""
    This isn't just a random guess. The AI uses advanced mathematics and two "brains" to make a decision.
    Here is the breakdown of the technology.
    """)
    
    st.subheader("1. The Two Experts")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🧠 Expert A: The Pattern Spotter (TCN)**")
        st.caption("Temporal Convolutional Network")
        st.write("This part of the brain looks at charts. It ignores news and focuses purely on math: *'If the price did X yesterday, it usually does Y today.'*")
    with col2:
        st.markdown("**🧠 Expert B: The Reader (Transformer)**")
        st.caption("Sentiment Analysis Engine")
        st.write("This part reads the news. It uses a model called 'DistilRoBERTa' to understand if headlines are fearful or happy.")

    st.subheader("2. The Math: Filling the Gaps")
    st.markdown("We need 5 years of history to train the AI, but news data is hard to find for the past. So, we use math to 'Imply' what the news probably was.")
    
    st.markdown("**The Implied Sentiment Formula:**")
    st.latex(r'''
    S_{implied} = \frac{RSI_{norm} + (1 - VIX_{norm})}{2}
    ''')
    st.markdown("""
    * **RSI:** Measures if people are buying too much.
    * **VIX:** Measures how scared people are.
    * **Logic:** If people are Scared (High VIX) and Selling (Low RSI), the news was probably bad.
    """)

    st.subheader("3. The Math: Detecting Panic (Z-Score)")
    st.markdown("How does the AI know if today is 'Normal' or 'Crazy'? It calculates a **Z-Score**.")
    st.latex(r'''
    Z = \frac{X - \mu}{\sigma}
    ''')
    st.markdown("""
    * $X$ = Today's Sentiment Score
    * $\mu$ = The Average Score of the last year
    * $\sigma$ = The Volatility (Standard Deviation)
    * **Translation:** If $Z$ is below -1.5, it means the mood is historically bad (Panic).
    """)

    st.subheader("4. Realism Check (Why we don't have a 'Neutral' Zone)")
    st.markdown("""
    You might ask: *"Why doesn't the AI just say 'I don't know'?"*
    
    We tested that! We added a "Neutral Zone" where the AI would sit in cash if it wasn't sure. 
    **It lost money.**
    
    Why? **Transaction Costs.**
    Every time you buy or sell, you pay a tiny fee (spread/slippage). If the AI is constantly getting in and out because it's "unsure," those fees add up.
    
    **Our Strategy:** It is mathematically better to stay in the market ("Always-In") unless there is a major crash signal.
    """)

# --- TAB 3: SETTINGS & RETRAIN ---
with tab3:
    st.header("🔧 Tweaks")
    
    st.markdown("### 1. Manual Override")
    sentiment_mode = st.radio("Where should the AI get sentiment?", ["Auto-Scrape News", "Manual Input Slider"])
    
    manual_val = 0.0
    if sentiment_mode == "Manual Input Slider":
        manual_val = st.slider("Set Sentiment (-1.0 is Panic, 1.0 is Euphoria)", -1.0, 1.0, 0.0, 0.1)
    
    st.markdown("### 2. Active Learning")
    st.write("Click this once a week to teach the AI using the latest data.")
    
    if st.button("Retrain AI Brain"):
        curr_ts = time.time()
        time_since_retrain = curr_ts - st.session_state['last_retrain']
        if time_since_retrain < RETRAIN_COOLDOWN_SECONDS:
            st.error(f"Please wait {int((RETRAIN_COOLDOWN_SECONDS - time_since_retrain)/60)} minutes.")
        else:
            st.session_state['last_retrain'] = curr_ts
            # (Retraining logic same as before...)
            model, scaler, config = load_resources()
            if model is None:
                st.error("Model missing.")
            else:
                with st.spinner("🧠 Teaching the AI..."):
                    # Quick fine-tune mock-up
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
                        st.success("✅ Brain Updated Successfully!")
                    except Exception as e:
                        st.error(f"Error: {e}")

st.markdown("---")
st.caption("Disclaimer: This is an educational tool demonstrating AI concepts. Not financial advice.")
