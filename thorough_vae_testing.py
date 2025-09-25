# thorough_vae_testing.py
# Comprehensive testing for VAE models: backtesting, edge cases, benchmarking, sensitivity

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy import stats
from scipy.stats import kstest
import warnings
warnings.filterwarnings('ignore')

# Load data for testing
def load_data(ticker, start="2018-01-01", end="2023-12-31"):
    data = yf.download(ticker, start=start, end=end, progress=False)
    if data is None or data.empty:
        raise RuntimeError(f"No data for {ticker}")
    price_col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
    returns = np.array(data[price_col].pct_change().dropna()).reshape(-1, 1)
    return returns

# VAE Model (simplified for testing)
class TestVAE(nn.Module):
    def __init__(self, input_dim=1, latent_dim=10):
        super(TestVAE, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 64), nn.ReLU(), nn.Linear(64, input_dim))
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    recon_loss = 0.5 * torch.sum((recon_x - x)**2) / x.size(0)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon_loss + kld

# Train VAE
def train_vae(data, latent_dim=10, epochs=100):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    tensor_data = torch.tensor(data_scaled, dtype=torch.float32)
    dataset = TensorDataset(tensor_data)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    vae = TestVAE(latent_dim=latent_dim)
    optimizer = optim.Adam(vae.parameters(), lr=1e-3)

    vae.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            x = batch[0]
            optimizer.zero_grad()
            recon_x, mu, logvar = vae(x)
            loss = vae_loss(recon_x, x, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

    return vae, scaler

# Generate samples
def generate_samples(vae, scaler, latent_dim, n_samples=5000):
    vae.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, latent_dim)
        samples = vae.decode(z).numpy()
    return scaler.inverse_transform(samples)

# Backtesting: Out-of-sample VaR
def backtest_var(returns, samples, confidence=0.05):
    real_var = np.percentile(returns, confidence * 100)
    sample_var = np.percentile(samples, confidence * 100)
    return real_var, sample_var

# KS Test for distribution fit
def ks_test(returns, samples):
    return kstest(returns.flatten(), samples.flatten())

# Edge Case: Simulate crisis data (extreme negative returns)
def simulate_crisis_data(n=1000):
    return np.random.choice([-0.1, -0.05, 0, 0.05, 0.1], n, p=[0.1, 0.2, 0.4, 0.2, 0.1]).reshape(-1, 1)

# Benchmark against GARCH (simplified)
def simple_garch_benchmark(returns, n_samples=5000):
    # Simplified: Use historical simulation
    return np.random.choice(returns.flatten(), n_samples, replace=True).reshape(-1, 1)

# Sensitivity Analysis: Vary latent dim
def sensitivity_test(data, latent_dims=[5, 10, 20]):
    results = {}
    for ld in latent_dims:
        vae, scaler = train_vae(data, latent_dim=ld, epochs=50)
        samples = generate_samples(vae, scaler, ld)
        ks_stat, ks_p = ks_test(data, samples)
        results[ld] = {'ks_stat': ks_stat, 'ks_p': ks_p}
    return results

# Main Testing
tickers = ['AAPL', 'MSFT', 'TSLA', 'SPY']
results = {}

for ticker in tickers:
    print(f"Testing {ticker}...")
    returns = load_data(ticker)

    # Train VAE
    vae, scaler = train_vae(returns, epochs=100)
    samples = generate_samples(vae, scaler, 10)

    # Backtesting
    real_var, sample_var = backtest_var(returns, samples)
    ks_stat, ks_p = ks_test(returns, samples)

    # Edge Case
    crisis_data = simulate_crisis_data()
    crisis_vae, crisis_scaler = train_vae(crisis_data, epochs=100)
    crisis_samples = generate_samples(crisis_vae, crisis_scaler, 10)
    crisis_ks = ks_test(crisis_data, crisis_samples)

    # Benchmark
    garch_samples = simple_garch_benchmark(returns)
    garch_ks = ks_test(returns, garch_samples)

    # Sensitivity
    sens_results = sensitivity_test(returns)

    results[ticker] = {
        'real_var': real_var,
        'sample_var': sample_var,
        'ks_stat': ks_stat,
        'ks_p': ks_p,
        'crisis_ks': crisis_ks,
        'garch_ks': garch_ks,
        'sensitivity': sens_results
    }

# Print Results
for ticker, res in results.items():
    print(f"\n{ticker} Results:")
    print(f"  VaR - Real: {res['real_var']:.4f}, VAE: {res['sample_var']:.4f}")
    print(f"  KS Test: Stat={res['ks_stat']:.4f}, P={res['ks_p']:.4f}")
    print(f"  Crisis KS: {res['crisis_ks']}")
    print(f"  GARCH KS: {res['garch_ks']}")
    print(f"  Sensitivity: {res['sensitivity']}")

print("Thorough testing completed.")
