# hybrid_model.py
# Hybrid approach: VAE for bulk generation + Historical simulation for tails

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import RobustScaler
from scipy.stats import kstest
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# 1. Load Dataset
# =========================
print("Downloading stock data...")
data = yf.download("AAPL", start="2018-01-01", end="2023-12-31")

if data is None or data.empty:
    raise RuntimeError("Failed to download data or data is empty")

if 'Adj Close' in data.columns:
    price_col = 'Adj Close'
elif 'Close' in data.columns:
    price_col = 'Close'
else:
    price_col = data.columns[0] if len(data.columns) > 0 else 'Close'

returns = np.array(data[price_col].pct_change().dropna()).reshape(-1, 1)

scaler = RobustScaler()
returns_scaled = scaler.fit_transform(returns)

tensor_data = torch.tensor(returns_scaled, dtype=torch.float32)
dataset = TensorDataset(tensor_data)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# =========================
# 2. Train VAE for Bulk Generation
# =========================
class HybridVAE(nn.Module):
    def __init__(self, input_dim=1, latent_dim=20):
        super(HybridVAE, self).__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

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

vae = HybridVAE(input_dim=1, latent_dim=20)
optimizer = optim.Adam(vae.parameters(), lr=1e-3)

epochs = 200
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
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

# =========================
# 3. Hybrid Sampling: VAE + Historical Tails
# =========================
def hybrid_sample(vae, scaler, historical_returns, n_samples=10000, tail_threshold=0.05):
    vae.eval()

    # Generate bulk samples with VAE
    with torch.no_grad():
        z = torch.randn(n_samples, vae.latent_dim)
        vae_samples_scaled = vae.decode(z).numpy()

    vae_samples = scaler.inverse_transform(vae_samples_scaled)

    # Identify extreme historical returns for tail replacement
    sorted_returns = np.sort(historical_returns.flatten())
    lower_tail = sorted_returns[:int(len(sorted_returns) * tail_threshold)]
    upper_tail = sorted_returns[-int(len(sorted_returns) * tail_threshold):]

    # Replace VAE samples in tails with historical extremes
    hybrid_samples = vae_samples.copy()

    # Replace lower tail
    lower_mask = vae_samples.flatten() < np.percentile(vae_samples, tail_threshold * 100)
    if np.sum(lower_mask) > 0:
        replacement_indices = np.random.choice(len(lower_tail), size=np.sum(lower_mask), replace=True)
        hybrid_samples[lower_mask, 0] = lower_tail[replacement_indices]

    # Replace upper tail
    upper_mask = vae_samples.flatten() > np.percentile(vae_samples, (1 - tail_threshold) * 100)
    if np.sum(upper_mask) > 0:
        replacement_indices = np.random.choice(len(upper_tail), size=np.sum(upper_mask), replace=True)
        hybrid_samples[upper_mask, 0] = upper_tail[replacement_indices]

    return hybrid_samples

# Generate hybrid samples
hybrid_samples = hybrid_sample(vae, scaler, returns, n_samples=10000, tail_threshold=0.05)
np.save("outputs/hybrid_samples.npy", hybrid_samples)
print("✅ Hybrid samples saved as hybrid_samples.npy")

# =========================
# 4. Evaluation
# =========================
real_var = np.percentile(returns, 5)
hybrid_var = np.percentile(hybrid_samples, 5)
print(f"Real VaR: {real_var:.4f}, Hybrid VaR: {hybrid_var:.4f}, Diff: {abs(real_var - hybrid_var):.4f}")

# KS test
ks_stat, ks_p = kstest(returns.flatten(), hybrid_samples.flatten())
print(f"KS Test: Stat={ks_stat:.4f}, P={ks_p:.4f}")

# =========================
# 5. Plot Comparison
# =========================
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.histplot(returns, bins=50, color="blue", label="Real Data", kde=True, stat="density")
sns.histplot(hybrid_samples, bins=50, color="red", label="Hybrid Samples", kde=True, stat="density", alpha=0.6)
plt.legend()
plt.title("Real vs Hybrid Learned Distribution")

plt.subplot(1, 2, 2)
# Zoom on tails
plt.hist(returns.flatten(), bins=100, alpha=0.5, label="Real", density=True)
plt.hist(hybrid_samples.flatten(), bins=100, alpha=0.5, label="Hybrid", density=True)
plt.xlim(-0.1, 0.1)  # Focus on tails
plt.legend()
plt.title("Tail Comparison (Zoomed)")

plt.tight_layout()
plt.savefig("outputs/hybrid_comparison.png")
plt.show()
