# temporal_robustness_testing.py
# Test VAE models on different market periods for temporal robustness

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

# Define market periods
periods = {
    "Calm Market (2017)": ("2017-01-01", "2017-12-31"),
    "Volatile Market (2020)": ("2020-02-01", "2020-04-30"),  # COVID crash
    "Recent Volatile (2022)": ("2022-01-01", "2022-12-31"),
    "Bull Market (2021)": ("2021-01-01", "2021-12-31")
}

tickers = ["AAPL", "SPY"]

def load_returns(ticker, start, end):
    data = yf.download(ticker, start=start, end=end, progress=False)
    if data is None or data.empty:
        raise RuntimeError(f"No data for {ticker} {start}-{end}")
    price_col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
    returns = np.array(data[price_col].pct_change().dropna()).reshape(-1, 1)
    return returns

def train_vae(returns, epochs=100):
    scaler = StandardScaler()
    returns_scaled = scaler.fit_transform(returns)
    tensor_data = torch.tensor(returns_scaled, dtype=torch.float32)
    dataset = TensorDataset(tensor_data)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Simple VAE for testing
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
    results = []
    for ticker in tickers:
        for period_name, (start, end) in periods.items():
            print(f"\nTesting {ticker} in {period_name} ({start} to {end})")
            try:
                returns = load_returns(ticker, start, end)
                samples = train_vae(returns, epochs=100)

                # Calculate VaR
                real_var = np.percentile(returns, 5)
                synth_var = np.percentile(samples, 5)

                # KS test
                ks_stat, ks_p = kstest(returns.flatten(), samples.flatten())

                results.append({
                    "Ticker": ticker,
                    "Period": period_name,
                    "Real VaR": real_var,
                    "Synth VaR": synth_var,
                    "VaR Diff": abs(real_var - synth_var),
                    "KS Stat": ks_stat,
                    "KS P": ks_p
                })

                print(f"Real VaR: {real_var:.4f}, Synth VaR: {synth_var:.4f}, KS p: {ks_p:.4f}")

            except Exception as e:
                print(f"Error for {ticker} {period_name}: {e}")

    # Summary
    df = pd.DataFrame(results)
    print("\nTemporal Robustness Results:")
    print(df)
    df.to_csv("outputs/temporal_robustness_results.csv", index=False)

    # Plot VaR differences
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="Period", y="VaR Diff", hue="Ticker")
    plt.title("VaR Estimation Error Across Market Periods")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("outputs/temporal_var_comparison.png")
    plt.show()

if __name__ == "__main__":
    main()
