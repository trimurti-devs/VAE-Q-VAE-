# Beginner-Friendly Demo: What This Project Does
# ===============================================
#
# This script shows in simple terms what our VAE project is solving.
# Imagine you have stock prices that go up and down. The "distribution" is the pattern of those ups and downs.
# Our goal: Teach a computer (VAE) to learn that pattern and create fake but realistic stock movements.
#
# Why? So we can test financial strategies without real money, or detect weird market behavior.
#
# This demo:
# 1. Gets real stock data (AAPL prices)
# 2. Trains a simple VAE to learn the pattern
# 3. Generates fake data that looks similar
# 4. Shows a comparison plot
#
# No advanced math needed - just run and see!

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

print("🚀 Starting Beginner Demo: Learning Stock Patterns with AI")
print("=" * 60)

# Step 1: Get Real Stock Data
print("📈 Step 1: Downloading real stock data (AAPL for 2023)")
data = yf.download("AAPL", start="2023-01-01", end="2023-12-31", progress=False)

# Find the price column
if 'Adj Close' in data.columns:
    price_col = 'Adj Close'
elif 'Close' in data.columns:
    price_col = 'Close'
else:
    price_col = data.columns[0]

# Calculate daily returns (percentage changes)
returns = data[price_col].pct_change().dropna().values.reshape(-1, 1)
print(f"   Got {len(returns)} days of stock returns")

# Step 2: Prepare Data for AI
print("🔧 Step 2: Preparing data for the AI model")
scaler = StandardScaler()
returns_scaled = scaler.fit_transform(returns)  # Make data easier for AI to learn

# Convert to PyTorch format
tensor_data = torch.tensor(returns_scaled, dtype=torch.float32)
dataset = TensorDataset(tensor_data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Step 3: Define the VAE Model
print("🤖 Step 3: Creating the AI model (VAE)")

class SimpleVAE(nn.Module):
    def __init__(self):
        super(SimpleVAE, self).__init__()

        # Encoder: Compresses data into a small "latent" space
        self.encoder = nn.Sequential(
            nn.Linear(1, 64),  # Input: 1 number (daily return)
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.mu = nn.Linear(32, 8)      # Mean of latent space
        self.logvar = nn.Linear(32, 8)  # Variance of latent space

        # Decoder: Reconstructs data from latent space
        self.decoder = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Output: 1 number (reconstructed return)
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

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

vae = SimpleVAE()

# Step 4: Train the VAE
print("🎓 Step 4: Training the AI (this may take a minute)")

def vae_loss(recon_x, x, mu, logvar):
    # Reconstruction loss: How well does it recreate the input?
    recon_loss = 0.5 * torch.sum((recon_x - x)**2) / x.size(0)

    # KL divergence: Keeps latent space normal (regularizes)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

    return recon_loss + kld

optimizer = optim.Adam(vae.parameters(), lr=0.001)
epochs = 50

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

    if (epoch + 1) % 10 == 0:
        print(f"   Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

print("   Training complete!")

# Step 5: Generate Fake Data
print("🎲 Step 5: Generating fake stock returns")
vae.eval()
with torch.no_grad():
    # Sample from latent space (like picking random ideas)
    z = torch.randn(1000, 8)  # 1000 fake samples
    fake_returns_scaled = vae.decode(z).numpy()

# Convert back to original scale
fake_returns = scaler.inverse_transform(fake_returns_scaled)

# Step 6: Compare Real vs Fake
print("📊 Step 6: Comparing real vs AI-generated data")

plt.figure(figsize=(10, 6))
sns.histplot(returns.flatten(), bins=30, color="blue", alpha=0.6, label="Real Stock Returns", kde=True, stat="density")
sns.histplot(fake_returns.flatten(), bins=30, color="red", alpha=0.6, label="AI-Generated Returns", kde=True, stat="density")
plt.title("Real vs AI-Generated Stock Return Patterns")
plt.xlabel("Daily Return")
plt.ylabel("Density")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("✅ Demo Complete!")
print("What you saw:")
print("- Blue: Real stock ups/downs pattern")
print("- Red: AI's attempt to copy that pattern")
print("- If they look similar, the AI learned well!")
print("- This is useful for creating 'what-if' scenarios in finance")
print("=" * 60)
