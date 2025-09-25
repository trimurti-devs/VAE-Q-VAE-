# risk_metrics_analysis.py
# Analyze risk metrics (VaR, Expected Shortfall) using synthetic samples from VAEs

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def value_at_risk(returns, alpha=0.05):
    """Calculate Value at Risk (VaR) at confidence level alpha"""
    return np.percentile(returns, 100 * alpha)

def expected_shortfall(returns, alpha=0.05):
    """Calculate Expected Shortfall (Conditional VaR) at confidence level alpha"""
    var = value_at_risk(returns, alpha)
    return returns[returns <= var].mean()

def analyze_risk_metrics(real_returns, samples_dict, alpha=0.05):
    """
    real_returns: np.array of real returns
    samples_dict: dict of {model_name: np.array of synthetic returns}
    alpha: confidence level for VaR and ES
    """
    results = []
    for model_name, samples in samples_dict.items():
        var = value_at_risk(samples, alpha)
        es = expected_shortfall(samples, alpha)
        results.append({
            "Model": model_name,
            "VaR": var,
            "Expected Shortfall": es
        })
    # Also compute for real returns
    real_var = value_at_risk(real_returns, alpha)
    real_es = expected_shortfall(real_returns, alpha)
    results.append({
        "Model": "Real Data",
        "VaR": real_var,
        "Expected Shortfall": real_es
    })
    return pd.DataFrame(results)

def print_tail_risk(samples_dict, threshold=-0.05):
    print(f"Tail Risk Probability (P[return < {threshold*100:.1f}%]):")
    for model_name, samples in samples_dict.items():
        prob = np.mean(samples < threshold)
        print(f"  {model_name}: {prob:.4%}")

def plot_risk_metrics(df):
    df_melt = df.melt(id_vars=["Model"], value_vars=["VaR", "Expected Shortfall"],
                      var_name="Metric", value_name="Value")
    plt.figure(figsize=(10,6))
    import seaborn as sns
    sns.barplot(data=df_melt, x="Metric", y="Value", hue="Model")
    plt.title("Risk Metrics Comparison (5% level)")
    plt.tight_layout()
    plt.savefig("outputs/risk_metrics_comparison.png")
    plt.show()

if __name__ == "__main__":
    import yfinance as yf

    # Load real returns for AAPL 2018
    data = yf.download("AAPL", start="2018-01-01", end="2018-12-31", progress=False)
    if data is None or data.empty:
        raise RuntimeError("Failed to download data or data is empty")
    if 'Adj Close' in data.columns:
        price_col = 'Adj Close'
    elif 'Close' in data.columns:
        price_col = 'Close'
    else:
        price_col = data.columns[0]
    real_returns = data[price_col].pct_change().dropna().values

    # Load synthetic samples
    vae_samples = np.load("outputs/vae_samples.npy").flatten()
    optimized_samples = np.load("outputs/optimized_vae_samples.npy").flatten()
    qvae_samples = np.load("outputs/q_vae_samples.npy").flatten()

    samples_dict = {
        "Original VAE": vae_samples,
        "Optimized VAE": optimized_samples,
        "Q-VAE": qvae_samples
    }

    # Analyze risk metrics
    risk_df = analyze_risk_metrics(real_returns, samples_dict, alpha=0.05)
    print(risk_df)

    # Print tail risk probabilities
    print_tail_risk(samples_dict, threshold=-0.05)

    # Plot risk metrics
    plot_risk_metrics(risk_df)
