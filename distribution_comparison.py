import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from fitter import Fitter
from scipy import stats
from scipy.interpolate import interp1d
from scipy.spatial.distance import jensenshannon

# ===============================
# 1. Load Real Data
# ===============================
data = yf.download("AAPL", start="2023-01-01", end="2023-12-31")

# Handle different column naming conventions in yfinance
if 'Adj Close' in data.columns:
    price_col = 'Adj Close'
elif 'Close' in data.columns:
    price_col = 'Close'
else:
    price_col = data.columns[0] if len(data.columns) > 0 else 'Close'

returns = data[price_col].pct_change().dropna().values

# ===============================
# 2. Load VAE Samples (original, optimized, Q-VAE)
# ===============================
vae_samples = np.load("D:/python/statistics/outputs/vae_samples.npy")
optimized_vae_samples = np.load("D:/python/statistics/outputs/optimized_vae_samples.npy")
q_vae_samples = np.load("D:/python/statistics/outputs/q_vae_samples.npy")

# ===============================
# 3. Fit Classical Distributions
# ===============================
f = Fitter(returns, distributions=["norm", "t", "laplace", "cauchy", "gamma", "lognorm"])
f.fit()
best_dist = f.get_best()
print("Best Classical Fit:", best_dist)

dist_name = list(best_dist.keys())[0]
params = best_dist[dist_name]

# Build scipy distribution
if dist_name == "t":
    dist = stats.t(df=params["df"], loc=params["loc"], scale=params["scale"])
elif dist_name == "norm":
    dist = stats.norm(loc=params["loc"], scale=params["scale"])
elif dist_name == "laplace":
    dist = stats.laplace(loc=params["loc"], scale=params["scale"])
elif dist_name == "cauchy":
    dist = stats.cauchy(loc=params["loc"], scale=params["scale"])
elif dist_name == "gamma":
    dist = stats.gamma(a=params["a"], loc=params["loc"], scale=params["scale"])
elif dist_name == "lognorm":
    dist = stats.lognorm(s=params["s"], loc=params["loc"], scale=params["scale"])
else:
    raise ValueError("Unsupported distribution: " + dist_name)

# ===============================
# 4. Statistical Tests
# ===============================
ks_classical = stats.kstest(returns, dist.cdf)

def empirical_cdf(data):
    """Create empirical CDF function from data"""
    data_sorted = np.sort(data)
    n = len(data)
    y = np.arange(1, n+1) / n
    return interp1d(data_sorted, y, kind='linear', bounds_error=False, fill_value=(0, 1))

ks_vae = stats.kstest(returns, empirical_cdf(vae_samples.flatten()))
ks_optimized_vae = stats.kstest(returns, empirical_cdf(optimized_vae_samples.flatten()))
ks_q_vae = stats.kstest(returns, empirical_cdf(q_vae_samples.flatten()))

print("\nKS Test Results:")
print("Classical Fit:", ks_classical)
print("Original VAE Fit:", ks_vae)
print("Optimized VAE Fit:", ks_optimized_vae)
print("Q-VAE Fit:", ks_q_vae)

# ===============================
# 4b. KL and Jensen-Shannon Divergence
# ===============================
hist_real, bin_edges = np.histogram(returns, bins=100, density=True)
hist_vae, _ = np.histogram(vae_samples, bins=bin_edges, density=True)
hist_opt_vae, _ = np.histogram(optimized_vae_samples, bins=bin_edges, density=True)
hist_q_vae, _ = np.histogram(q_vae_samples, bins=bin_edges, density=True)

# Smooth and normalize
hist_real += 1e-8
hist_vae += 1e-8
hist_opt_vae += 1e-8
hist_q_vae += 1e-8
hist_real /= hist_real.sum()
hist_vae /= hist_vae.sum()
hist_opt_vae /= hist_opt_vae.sum()
hist_q_vae /= hist_q_vae.sum()

# KL divergence
kl_div_vae = np.sum(hist_real * np.log(hist_real / hist_vae))
kl_div_opt = np.sum(hist_real * np.log(hist_real / hist_opt_vae))
kl_div_q = np.sum(hist_real * np.log(hist_real / hist_q_vae))
print("\nKL Divergence:")
print("Real vs Original VAE:", kl_div_vae)
print("Real vs Optimized VAE:", kl_div_opt)
print("Real vs Q-VAE:", kl_div_q)

# JSD divergence (better metric, bounded [0,1])
jsd_vae = jensenshannon(hist_real, hist_vae)**2
jsd_opt = jensenshannon(hist_real, hist_opt_vae)**2
jsd_q = jensenshannon(hist_real, hist_q_vae)**2
print("\nJensen-Shannon Divergence:")
print("Real vs Original VAE:", jsd_vae)
print("Real vs Optimized VAE:", jsd_opt)
print("Real vs Q-VAE:", jsd_q)

# ===============================
# 5. Plot All Together
# ===============================
plt.figure(figsize=(12,8))

# Real data
sns.histplot(returns, bins=50, kde=True, stat="density", color="blue", alpha=0.4, label="Real Data")

# VAE samples
sns.histplot(vae_samples, bins=50, kde=True, stat="density", color="red", alpha=0.3, label="Original VAE Samples")
sns.histplot(optimized_vae_samples, bins=50, kde=True, stat="density", color="orange", alpha=0.3, label="Optimized VAE Samples")
sns.histplot(q_vae_samples, bins=50, kde=True, stat="density", color="purple", alpha=0.3, label="Q-VAE Samples")

# Classical distribution curve
x = np.linspace(min(returns), max(returns), 500)
plt.plot(x, dist.pdf(x), "g-", lw=2, label=f"Best Classical ({dist_name})")

plt.title("Real vs VAEs vs Classical Distribution (AAPL Returns)")
plt.xlabel("Returns")
plt.ylabel("Density")
plt.legend()
plt.show()
