# Hybrid Context-Aware AI Trader

**A Risk-First Algorithmic Trading System that combines Deep Learning (TCNs & Transformers) with Quantitative Logic (Z-Score Regime Detection) to predict S&P 500 trends.**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red) ![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B) ![License](https://img.shields.io/badge/License-MIT-green)

## Overview

Most AI trading models fail because they lack context—they treat a calm market and a crashing market identically. Furthermore, obtaining historical news sentiment usually requires expensive Bloomberg Terminal access.

This project addresses these limitations by implementing a **Context-Aware Ensemble**. It utilizes a **Gating Network** (Mixture of Experts) to dynamically switch strategies based on Market Volatility (VIX) and introduces a novel **"Implied Sentiment"** engine to reconstruct historical context without paid APIs.

The system integrates:

1.  **Technical Analysis:** A **Temporal Convolutional Network (TCN)** to track price momentum and historical trends.
2.  **Fundamental Analysis:** A **Transformer + VADER** pipeline that scrapes and analyzes live news headlines (via Google News RSS).
3.  **Statistical Safety:** A proprietary **Sentiment Z-Score** logic that detects "Black Swan" panic events by comparing current news sentiment against a market-implied historical baseline.

The result is a streamlined dashboard for retail investors that outputs clear **BUY** (Long/Safe) or **WAIT** (Cash/Defensive) signals.

## Key Features

* **Hybrid Architecture:** Merges a TCN (Time-Series) with a Transformer (NLP) via a learnable Gating Network.
* **Resilient News Pipeline:** Uses **Google News RSS** to fetch real-time headlines, bypassing the anti-bot blocking often found on standard financial sites like Finviz.
* **Implied Sentiment Engine:** Solves the "Paywall Problem" by reverse-engineering historical sentiment from VIX and RSI data. This creates a statistically valid Z-Score baseline without needing 20 years of paid news archives.
* **Active Learning (Retraining):** Includes a "Retrain AI" module that allows the model to fine-tune itself on the most recent 60 days of market data directly from the dashboard.
* **Volatility-Gated Logic:** The neural network learns to weigh technical factors during calm markets and sentiment factors during high-volatility events.
* **Anti-Spam Protection:** Built-in cooldown timers to prevent IP bans during the web scraping process.

## Architecture



The model follows a **Mixture of Experts (MoE)** design:

1.  **Input:** A 30-day lookback window containing Returns, VIX, Momentum, and Sentiment Z-Scores.
2.  **Expert A (Technicals):** A TCN with dilated causal convolutions extracts trend patterns from price data.
3.  **Expert B (Sentiment):** A Transformer Encoder processes the narrative structure of market fear and news sentiment.
4.  **Gating Network:** A Feed-Forward Network monitors the **VIX** (Volatility Index) and assigns weights to the experts in real-time.
    * *Low VIX* -> The model prioritizes Momentum/Technicals.
    * *High VIX* -> The model prioritizes Sentiment and Panic Signals.

## Installation & Usage

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/rAdvirtua/context-aware-ensemble.git](https://github.com/rAdvirtua/context-aware-ensemble.git)
    cd context-aware-ensemble
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Dashboard:**
    ```bash
    streamlit run app.py
    ```

4.  **Usage:**
    * **Live Analysis:** Click "Analyze Market Now" to trigger the live data fetch and inference pipeline.
    * **Retraining:** Open the sidebar and click "Retrain AI Brain" to fine-tune the model on the latest market data.

## File Structure

* `app.py`: The main Streamlit application containing the frontend, inference logic, and retraining module.
* `sp500_screener.py`: A standalone script to scan the Top 50 S&P 500 stocks for individual opportunities.
* `hybrid_model.pth`: Pre-trained PyTorch model weights.
* `scaler.pkl`: Scikit-learn scaler for data normalization.
* `model_config.json`: Hyperparameter configuration file.

## Disclaimer

**This is a research project, not financial advice.** The model outputs are probabilistic predictions based on historical data. Algorithmic trading involves significant risk of capital loss. The "Implied Sentiment" logic is a proxy approximation and may not reflect actual historical news events perfectly. Always perform your own due diligence before investing.
