# Hybrid Context-Aware AI Trader

**A Risk-First Algorithmic Trading System that combines Deep Learning (TCNs & Financial Transformers) with Quantitative Logic (Z-Score Regime Detection) to predict S&P 500 trends.**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red) ![Transformers](https://img.shields.io/badge/HuggingFace-DistilRoBERTa-yellow) ![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B) ![License](https://img.shields.io/badge/License-MIT-green)

## Overview

Most AI trading models fail because they lack context—they treat a calm market and a crashing market identically. Furthermore, obtaining historical news sentiment usually requires expensive Bloomberg Terminal access.

This project addresses these limitations by implementing a **Context-Aware Ensemble**. It utilizes a **Gating Network** (Mixture of Experts) to dynamically switch strategies based on Market Volatility (VIX) and introduces a novel **"Data Flywheel"** architecture to build a proprietary dataset over time.

The system integrates:

1.  **Technical Analysis:** A **Temporal Convolutional Network (TCN)** to track price momentum and historical trends.
2.  **Fundamental Analysis:** A **DistilRoBERTa Financial Transformer** (NLP) that replaces VADER to understand complex financial nuance (e.g., "Deficit narrowed" is positive).
3.  **The "Time-Bridge":** A hybrid data engine that stitches together real historical data (2018-2024), implied proxy data (gaps), and live scraped data (today) to create a seamless training timeline.

The result is a streamlined dashboard for retail investors that outputs clear **BUY** (Long/Safe) or **WAIT** (Cash/Defensive) signals.

## Key Features

* **Hybrid Architecture:** Merges a TCN (Time-Series) with a Transformer (NLP) via a learnable Gating Network.
* **Financial BERT Engine:** Uses `mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis` for state-of-the-art financial text classification, offering significantly higher accuracy than dictionary models like VADER.
* **The Data Moat:** Automatically logs every analysis run (Headlines + Sentiment + Market VIX) into a local `proprietary_dataset.csv`. This creates a unique, high-value dataset that grows every time you use the app, allowing for future fine-tuning.
* **Implied Sentiment Engine:** Solves the "Paywall Problem" by reverse-engineering historical sentiment from VIX and RSI data for periods where news archives are unavailable.
* **Active Learning (Retraining):** Includes a "Retrain AI" module that allows the model to fine-tune itself on the most recent 60 days of market data directly from the dashboard.

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
    * **Live Analysis:** Click "Analyze Market Now" to trigger the live data fetch (Google News), NLP inference (DistilRoBERTa), and TCN prediction.
    * **Data Harvesting:** The app will automatically create and append data to `proprietary_dataset.csv`.
    * **Retraining:** Open the sidebar and click "Retrain AI Brain" to fine-tune the model on the latest market data.

## File Structure

* `app.py`: The main Streamlit application containing the frontend, DistilRoBERTa pipeline, and inference logic.
* `hybrid_model.pth`: Pre-trained PyTorch model weights.
* `scaler.pkl`: Scikit-learn scaler for data normalization.
* `model_config.json`: Hyperparameter configuration file.
* `mcwsi_historical_2024.csv`: The base historical dataset (2018-2024).
* `proprietary_dataset.csv`: **[AUTO-GENERATED]** The app creates this file to store your unique dataset.

## Disclaimer

**This is a research project, not financial advice.** The model outputs are probabilistic predictions based on historical data. Algorithmic trading involves significant risk of capital loss. The "Implied Sentiment" logic is a proxy approximation and may not reflect actual historical news events perfectly. Always perform your own due diligence before investing.
