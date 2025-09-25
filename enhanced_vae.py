# enhanced_vae.py
# Enhanced VAE with heavy-tailed priors and outlier-aware loss for better tail risk modeling

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import RobustScaler
from scipy.stats import kstest, t
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

scaler = RobustScaler()  # Better for outliers
returns_scaled = scaler.fit_transform(returns)

tensor_data = torch.tensor(returns_scaled, dtype=torch.float32)
dataset = TensorDataset(tensor_data)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# =========================
# 2. Enhanced VAE with Heavy-Tailed Latent Space
# =========================
class EnhancedVAE(nn.Module):
    def __init__(self, input_dim=1, latent_dim=20):
        super(EnhancedVAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        # Use heavy-tailed sampling: mixture of Gaussians
        std = torch.exp(0.5 * logvar)
        # Sample from normal with occasional heavy tails
        z_normal = torch.randn_like(std)
        # Add occasional large deviations
        heavy_tail = torch.randn_like(std) * 3.0  # Larger variance
        mask = torch.rand_like(std) < 0.1  # 10% heavy tail samples
        z = torch.where(mask, heavy_tail, z_normal)
        z = mu + std * z
        return z

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# =========================
# 3. Outlier-Aware Loss Function
# =========================
def outlier_aware_loss(recon_x, x, mu, logvar, outlier_threshold=2.0):
    # Standard reconstruction loss with outlier weighting
    errors = torch.abs(recon_x - x)
    weights = torch.where(errors > outlier_threshold, 2.0, 1.0)  # Weight outliers more
    recon_loss = torch.mean(weights * errors ** 2)

    # KL divergence
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

    return recon_loss + kld

# =========================
# 4. Training
# =========================
vae = EnhancedVAE(input_dim=1, latent_dim=20)
optimizer = optim.Adam(vae.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)

epochs = 200
vae.train()
for epoch in range(epochs):
    total_loss = 0
    for batch in dataloader:
        x = batch[0]
        optimizer.zero_grad()
        recon_x, mu, logvar = vae(x)
        loss = outlier_aware_loss(recon_x, x, mu, logvar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

# Save model
torch.save(vae.state_dict(), "outputs/enhanced_vae.pth")

# =========================
# 5. Sampling and Evaluation
# =========================
vae.eval()
with torch.no_grad():
    z = torch.randn(10000, vae.latent_dim)
    samples = vae.decode(z).numpy()

samples = scaler.inverse_transform(samples)
np.save("outputs/enhanced_vae_samples.npy", samples)
print("✅ Enhanced VAE samples saved as enhanced_vae_samples.npy")

# Calculate VaR
real_var = np.percentile(returns, 5)
synth_var = np.percentile(samples, 5)
print(f"Real VaR: {real_var:.4f}, Synth VaR: {synth_var:.4f}, Diff: {abs(real_var - synth_var):.4f}")

# KS test
ks_stat, ks_p = kstest(returns.flatten(), samples.flatten())
print(f"KS Test: Stat={ks_stat:.4f}, P={ks_p:.4f}")

# =========================
# 6. Plot Comparison
# =========================
plt.figure(figsize=(10, 6))
sns.histplot(returns, bins=50, color="blue", label="Real Data", kde=True, stat="density")
sns.histplot(samples, bins=50, color="green", label="Enhanced VAE Samples", kde=True, stat="density", alpha=0.6)
plt.legend()
plt.title("Real vs Enhanced VAE Learned Distribution (AAPL Returns)")
plt.show()
