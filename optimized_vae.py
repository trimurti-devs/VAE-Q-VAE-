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

class OptimizedVAE(nn.Module):
    def __init__(self, input_dim=1, latent_dim=10):
        super(OptimizedVAE, self).__init__()

        # Deeper Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)

        # Deeper Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
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
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss(recon_x, x, mu, logvar, beta=1.0, kl_weight=1.0):
    # Gaussian NLL for reconstruction
    recon_loss = 0.5 * torch.sum((recon_x - x)**2) / x.size(0)  # Simplified Gaussian NLL assuming unit variance
    # KL divergence
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon_loss + beta * kl_weight * kld

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
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    vae = OptimizedVAE(input_dim=1, latent_dim=10)
    optimizer = optim.Adam(vae.parameters(), lr=1e-3)

    epochs = 200
    beta = 0.1  # Beta-VAE for disentanglement
    vae.train()
    print(f"Training Optimized VAE for {args.symbol}...")
    for epoch in range(epochs):
        total_loss = 0
        kl_w = min(1.0, 0.1 + 0.9 * epoch / epochs)  # KL annealing from 0.1 to 1.0
        for batch in dataloader:
            x = batch[0]
            optimizer.zero_grad()
            recon_x, mu, logvar = vae(x)
            loss = vae_loss(recon_x, x, mu, logvar, beta=beta, kl_weight=kl_w)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}, KL Weight: {kl_w:.3f}")

    vae.eval()
    with torch.no_grad():
        z = torch.randn(10000, 10)
        samples = vae.decode(z).numpy()

    samples = scaler.inverse_transform(samples)
    os.makedirs("outputs", exist_ok=True)
    np.save(f"outputs/optimized_vae_samples_{args.symbol}.npy", samples)
    print(f"✅ Optimized VAE samples saved as outputs/optimized_vae_samples_{args.symbol}.npy")

    # Plot Comparison
    plt.figure(figsize=(10, 6))
    sns.histplot(returns, bins=50, color="blue", label="Real Data", kde=True, stat="density")
    sns.histplot(samples, bins=50, color="red", label="Optimized VAE Samples", kde=True, stat="density", alpha=0.6)
    plt.legend()
    plt.title(f"Real vs Optimized VAE Learned Distribution ({args.symbol} Returns)")
    plt.show()

if __name__ == "__main__":
    main()
