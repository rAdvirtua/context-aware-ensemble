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

## Mathematical Foundation

The system relies on three core mathematical innovations to handle data gaps and regime changes.

### 1. Z-Score Normalization (The "MCWSI Proxy")
Raw sentiment scores from the NLP model are volatile and lack historical context. A score of `0.5` might be bullish in a bear market but neutral in a bull market. We normalize the live score against a rolling 365-day window to derive a **Z-Score** ($Z_t$):

$$
Z_t = \frac{S_t - \mu_{t-365}}{\sigma_{t-365}}
$$

* Where $S_t$ is the raw sentiment score, $\mu$ is the rolling mean, and $\sigma$ is the rolling standard deviation.
* **Anomaly Detection:** If $Z_t < -2.0$, the system flags a statistically significant "Black Swan" panic event.

### 2. Implied Sentiment (The "Time Machine")
To train the model over long periods where news data is unavailable (e.g., data gaps between historical CSVs and live data), we infer sentiment ($S_{implied}$) from market observables:

$$
S_{implied} \approx \frac{\text{RSI}_{norm} + (1 - \text{VIX}_{norm})}{2}
$$

This assumes that high volatility ($VIX$) and low momentum ($RSI$) are valid mathematical proxies for negative news flow during data blackouts.

### 3. Mixture of Experts (Gating Logic)
The final decision $Y$ is a weighted sum of two "Expert" Neural Networks. The weights are determined by a **Gating Network** that watches the VIX:

$$
Y = w_{TCN} \cdot E_{TCN}(x) + w_{NLP} \cdot E_{NLP}(x)
$$

The gating weights $w$ are calculated via a Softmax function:

$$
w = \text{Softmax}(W_g \cdot \text{VIX} + b_g)
$$

* **Low VIX:** $w_{TCN} \to 1$ (The model trusts Price Trends).
* **High VIX:** $w_{NLP} \to 1$ (The model trusts News/Sentiment).

## Architecture

The model follows a **Mixture of Experts (MoE)** design:

1.  **Input:** A 30-day lookback window containing Returns, VIX, Momentum, and Sentiment Z-Scores.
2.  **Expert A (Technicals):** A TCN with dilated causal convolutions extracts trend patterns from price data.
3.  **Expert B (Sentiment):** A Transformer Encoder processes the narrative structure of market fear and news sentiment.
4.  **Gating Network:** A Feed-Forward Network monitors the **VIX** (Volatility Index) and assigns weights to the experts in real-time.

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
* `mcwsi_historical_2024.csv`: **[REQUIRED]** The base historical dataset (2008-2024) sourced from Dyuti Dasmahaptra's S&P 500 Financial News dataset.
* `proprietary_dataset.csv`: **[AUTO-GENERATED]** The app creates this file to store your unique dataset (Live News + Market Reactions).

## Data Sources & Credits

This project utilizes verified historical data to establish its sentiment baseline.

* **S&P 500 with Financial News Headlines (2008–2024):**
    * **Author:** Dyuti Dasmahaptra
    * **Source:** [Kaggle Dataset](https://www.kaggle.com/datasets/dyutidasmahaptra/s-and-p-500-with-financial-news-headlines-20082024)
    * **Usage:** This dataset provides the foundational "Ground Truth" for our sentiment analysis model (referenced as `mcwsi_historical_2024.csv`), allowing the system to understand market context prior to the live data stream.

## Disclaimer

**This is a research project, not financial advice.** The model outputs are probabilistic predictions based on historical data. Algorithmic trading involves significant risk of capital loss. The "Implied Sentiment" logic is a proxy approximation and may not reflect actual historical news events perfectly. Always perform your own due diligence before investing.
