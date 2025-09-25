"""
vae_distribution_learning.py

This script trains a Variational Autoencoder (VAE) to learn
the hidden probability distribution of real-world data
(stock returns in this case).

Steps:
1. Download stock return data.
2. Fit data into a VAE model.
3. Sample from latent space.
4. Compare learned distribution vs real data.
5. Save VAE samples for benchmarking.
"""

import argparse
import os
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

class VAE(nn.Module):
    def __init__(self, input_dim=1, latent_dim=2):
        super(VAE, self).__init__()
        
        # Encoder
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc_mu = nn.Linear(16, latent_dim)
        self.fc_logvar = nn.Linear(16, latent_dim)
        
        # Decoder
        self.fc2 = nn.Linear(latent_dim, 16)
        self.fc3 = nn.Linear(16, input_dim)

    def encode(self, x):
        h = torch.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = torch.relu(self.fc2(z))
        return self.fc3(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    recon_loss = nn.MSELoss()(recon_x, x)  # reconstruction error
    # KL divergence
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld / x.size(0)

def download_returns(symbol, start, end):
    data = yf.download(symbol, start=start, end=end, progress=False)
    if data is None or data.empty:
        raise RuntimeError(f"No data for {symbol} {start} to {end}")
    if 'Adj Close' in data.columns:
        price_col = 'Adj Close'
    elif 'Close' in data.columns:
        price_col = 'Close'
    else:
        price_col = data.columns[0]
    returns = data[price_col].pct_change().dropna().values.reshape(-1, 1)
    return returns

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="AAPL", help="Ticker symbol to download")
    parser.add_argument("--start", default="2018-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2023-12-31", help="End date YYYY-MM-DD")
    args = parser.parse_args()

    returns = download_returns(args.symbol, args.start, args.end)

    scaler = StandardScaler()
    returns_scaled = scaler.fit_transform(returns)

    tensor_data = torch.tensor(returns_scaled, dtype=torch.float32)
    dataset = TensorDataset(tensor_data)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    vae = VAE(input_dim=1, latent_dim=2)
    optimizer = optim.Adam(vae.parameters(), lr=0.001)

    epochs = 500
    vae.train()
    print(f"Training VAE for {args.symbol}...")
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

    vae.eval()
    with torch.no_grad():
        z = torch.randn(5000, 2)
        samples = vae.decode(z).numpy()

    samples = scaler.inverse_transform(samples)
    os.makedirs("outputs", exist_ok=True)
    np.save(f"outputs/vae_samples_{args.symbol}.npy", samples)
    print(f"✅ VAE samples saved as outputs/vae_samples_{args.symbol}.npy")

    # Plot Comparison
    plt.figure(figsize=(10, 6))
    sns.histplot(returns, bins=50, color="blue", label="Real Data", kde=True, stat="density")
    sns.histplot(samples, bins=50, color="red", label="VAE Samples", kde=True, stat="density", alpha=0.6)
    plt.legend()
    plt.title(f"Real vs VAE Learned Distribution ({args.symbol} Returns)")
    plt.show()

if __name__ == "__main__":
    main()
