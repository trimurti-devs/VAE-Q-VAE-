# visualization_and_analysis.py
# Script to visualize KL and JSD divergences across models and periods,
# and to compare synthetic return distributions for practical insights.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load extended testing results
df = pd.read_csv("outputs/extended_testing_results.csv")

# Melt dataframe for KL divergence visualization
kl_df = df.melt(id_vars=["ticker", "period"], 
                value_vars=["kl_original", "kl_optimized", "kl_qvae"],
                var_name="Model", value_name="KL Divergence")
kl_df["Model"] = kl_df["Model"].str.replace("kl_", "").str.capitalize()

# Plot KL Divergence by Model and Ticker
plt.figure(figsize=(12, 6))
sns.barplot(data=kl_df, x="ticker", y="KL Divergence", hue="Model")
plt.title("KL Divergence Across Models and Stocks")
plt.ylabel("KL Divergence")
plt.xlabel("Stock Ticker")
plt.legend(title="Model")
plt.tight_layout()
plt.savefig("outputs/kl_divergence_comparison.png")
plt.show()

# Melt dataframe for JSD visualization
jsd_df = df.melt(id_vars=["ticker", "period"], 
                 value_vars=["jsd_original", "jsd_optimized", "jsd_qvae"],
                 var_name="Model", value_name="JSD")
jsd_df["Model"] = jsd_df["Model"].str.replace("jsd_", "").str.capitalize()

# Plot JSD by Model and Ticker
plt.figure(figsize=(12, 6))
sns.barplot(data=jsd_df, x="ticker", y="JSD", hue="Model")
plt.title("Jensen-Shannon Divergence Across Models and Stocks")
plt.ylabel("Jensen-Shannon Divergence")
plt.xlabel("Stock Ticker")
plt.legend(title="Model")
plt.tight_layout()
plt.savefig("outputs/jsd_comparison.png")
plt.show()

# Practical Experiment: Overlay synthetic returns for a selected stock and period
import numpy as np

def load_samples(filename):
    return np.load(filename)

stock = "AAPL"
period = "2018-01-01 to 2018-12-31"

# Load real returns
import yfinance as yf
import numpy as np

def load_real_returns(ticker="AAPL", start="2018-01-01", end="2018-12-31"):
    data = yf.download(ticker, start=start, end=end, progress=False)
    if data is None or data.empty:
        raise RuntimeError(f"No data for {ticker} {start} to {end}")
    if 'Adj Close' in data.columns:
        price_col = 'Adj Close'
    elif 'Close' in data.columns:
        price_col = 'Close'
    else:
        price_col = data.columns[0]
    returns = np.array(data[price_col].pct_change().dropna())
    return returns

real_returns = load_real_returns("AAPL", "2018-01-01", "2018-12-31")

# Load synthetic samples
orig_samples = load_samples("outputs/vae_samples.npy")
opt_samples = load_samples("outputs/optimized_vae_samples.npy")
qvae_samples = load_samples("outputs/q_vae_samples.npy")

plt.figure(figsize=(10, 6))
sns.histplot(data=real_returns, bins=50, color="blue", label="Real Data", kde=True, stat="density")
sns.histplot(data=orig_samples.flatten(), bins=50, color="orange", label="Original VAE", kde=True, stat="density", alpha=0.6)
sns.histplot(data=opt_samples.flatten(), bins=50, color="green", label="Optimized VAE", kde=True, stat="density", alpha=0.6)
sns.histplot(data=qvae_samples.flatten(), bins=50, color="red", label="Q-VAE", kde=True, stat="density", alpha=0.6)
plt.legend()
plt.title(f"Synthetic vs Real Returns Distribution ({stock} {period})")
plt.savefig("outputs/synthetic_vs_real_distribution.png")
plt.show()

# Groundbreaking Idea: Risk Insights
# This is a conceptual placeholder for further research.
# You can analyze tail probabilities and extreme event frequencies
# using the synthetic samples to detect hidden risks.

print("Visualization and analysis complete. Check the outputs/ folder for plots.")
