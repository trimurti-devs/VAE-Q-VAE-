# multi_ticker_risk_analysis.py
# Run risk metrics and tail risk analysis across multiple tickers and periods

import yfinance as yf
import numpy as np
import pandas as pd
from risk_metrics_analysis import analyze_risk_metrics, print_tail_risk, plot_risk_metrics

tickers = ["AAPL", "MSFT", "TSLA", "SPY"]
start_date = "2018-01-01"
end_date = "2023-12-31"

all_results = []

for ticker in tickers:
    print(f"Processing {ticker}...")
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if data is None or data.empty:
        print(f"Warning: No data for {ticker} in {start_date} to {end_date}")
        continue
    if 'Adj Close' in data.columns:
        price_col = 'Adj Close'
    elif 'Close' in data.columns:
        price_col = 'Close'
    else:
        price_col = data.columns[0]
    real_returns = data[price_col].pct_change().dropna().values

    # Load synthetic samples for this ticker if available, else skip
    try:
        vae_samples = np.load(f"outputs/vae_samples_{ticker}.npy").flatten()
        optimized_samples = np.load(f"outputs/optimized_vae_samples_{ticker}.npy").flatten()
        qvae_samples = np.load(f"outputs/q_vae_samples_{ticker}.npy").flatten()
    except FileNotFoundError:
        print(f"Warning: Synthetic samples for {ticker} not found, skipping.")
        continue

    samples_dict = {
        "Original VAE": vae_samples,
        "Optimized VAE": optimized_samples,
        "Q-VAE": qvae_samples
    }

    risk_df = analyze_risk_metrics(real_returns, samples_dict, alpha=0.05)
    risk_df["Ticker"] = ticker
    all_results.append(risk_df)

    print_tail_risk(samples_dict, threshold=-0.05)

if all_results:
    combined_df = pd.concat(all_results, ignore_index=True)
    print(combined_df)
    plot_risk_metrics(combined_df)
else:
    print("No results to display. Please ensure synthetic samples are available for all tickers.")
