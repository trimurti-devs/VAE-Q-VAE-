# anomaly_injection_testing.py
# Test VAE robustness by injecting synthetic anomalies into data

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from scipy.stats import kstest
import matplotlib.pyplot as plt
import seaborn as sns

def load_returns(ticker="AAPL", start="2018-01-01", end="2018-12-31"):
    data = yf.download(ticker, start=start, end=end, progress=False)
    if data is None or data.empty:
        raise RuntimeError(f"No data for {ticker}")
    price_col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
    returns = np.array(data[price_col].pct_change().dropna()).reshape(-1, 1)
    return returns

def inject_anomalies(returns, anomaly_fraction=0.05, anomaly_magnitude=3.0):
    """Inject synthetic anomalies (extreme returns) into data"""
    n_anomalies = int(len(returns) * anomaly_fraction)
    anomaly_indices = np.random.choice(len(returns), n_anomalies, replace=False)

    anomalous_returns = returns.copy()
    for idx in anomaly_indices:
        # Inject extreme negative or positive return
        sign = np.random.choice([-1, 1])
        anomaly = sign * anomaly_magnitude * np.std(returns)
        anomalous_returns[idx] = anomaly

    return anomalous_returns

def train_vae(returns, epochs=100):
    scaler = StandardScaler()
    returns_scaled = scaler.fit_transform(returns)
    tensor_data = torch.tensor(returns_scaled, dtype=torch.float32)
    dataset = TensorDataset(tensor_data)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    class TestVAE(nn.Module):
        def __init__(self):
            super(TestVAE, self).__init__()
            self.encoder = nn.Sequential(nn.Linear(1, 64), nn.ReLU())
            self.decoder = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 1))
            self.fc_mu = nn.Linear(64, 10)
            self.fc_logvar = nn.Linear(64, 10)

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

    vae = TestVAE()
    optimizer = optim.Adam(vae.parameters(), lr=1e-3)

    vae.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            x = batch[0]
            optimizer.zero_grad()
            recon_x, mu, logvar = vae(x)
            recon_loss = 0.5 * torch.sum((recon_x - x)**2) / x.size(0)
            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
            loss = recon_loss + kld
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 25 == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

    # Generate samples
    vae.eval()
    with torch.no_grad():
        z = torch.randn(5000, 10)
        samples = vae.decode(z).numpy()
    samples = scaler.inverse_transform(samples)
    return samples

def main():
    ticker = "AAPL"
    returns = load_returns(ticker)

    # Test on clean data
    print("Testing on clean data...")
    clean_samples = train_vae(returns)
    clean_var = np.percentile(returns, 5)
    clean_synth_var = np.percentile(clean_samples, 5)
    clean_ks = kstest(returns.flatten(), clean_samples.flatten())

    # Test on anomalous data
    anomaly_levels = [0.05, 0.10, 0.20]  # 5%, 10%, 20% anomalies
    results = []

    for level in anomaly_levels:
        print(f"\nTesting with {level*100:.0f}% anomalies...")
        anomalous_returns = inject_anomalies(returns, anomaly_fraction=level)
        anomalous_samples = train_vae(anomalous_returns)

        real_var = np.percentile(anomalous_returns, 5)
        synth_var = np.percentile(anomalous_samples, 5)
        ks_stat, ks_p = kstest(anomalous_returns.flatten(), anomalous_samples.flatten())

        results.append({
            "Anomaly Level": f"{level*100:.0f}%",
            "Real VaR": real_var,
            "Synth VaR": synth_var,
            "VaR Diff": abs(real_var - synth_var),
            "KS Stat": ks_stat,
            "KS P": ks_p
        })

        print(f"Real VaR: {real_var:.4f}, Synth VaR: {synth_var:.4f}, KS p: {ks_p:.4f}")

    # Summary
    df = pd.DataFrame(results)
    print("\nAnomaly Injection Results:")
    print(df)
    df.to_csv("outputs/anomaly_injection_results.csv", index=False)

    # Plot comparison
    plt.figure(figsize=(12, 6))

    # Clean data
    plt.subplot(1, 2, 1)
    sns.histplot(returns.flatten(), bins=50, color="blue", alpha=0.6, label="Clean Real", kde=True, stat="density")
    sns.histplot(clean_samples.flatten(), bins=50, color="red", alpha=0.6, label="Clean Synth", kde=True, stat="density")
    plt.title("Clean Data Distribution")
    plt.legend()

    # Anomalous data (highest level)
    anomalous_returns = inject_anomalies(returns, anomaly_fraction=0.20)
    anomalous_samples = train_vae(anomalous_returns)
    plt.subplot(1, 2, 2)
    sns.histplot(anomalous_returns.flatten(), bins=50, color="green", alpha=0.6, label="Anomalous Real", kde=True, stat="density")
    sns.histplot(anomalous_samples.flatten(), bins=50, color="orange", alpha=0.6, label="Anomalous Synth", kde=True, stat="density")
    plt.title("Anomalous Data Distribution (20% anomalies)")
    plt.legend()

    plt.tight_layout()
    plt.savefig("outputs/anomaly_injection_comparison.png")
    plt.show()

if __name__ == "__main__":
    main()
