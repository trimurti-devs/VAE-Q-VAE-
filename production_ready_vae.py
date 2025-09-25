# production_ready_vae.py
# Production-ready VAE for financial risk modeling with improvements

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import RobustScaler
from scipy import stats
from scipy.stats import kstest, norm
import logging
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProductionVAE(nn.Module):
    def __init__(self, input_dim=1, latent_dim=20):
        super(ProductionVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )
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

def asymmetric_vae_loss(recon_x, x, mu, logvar, alpha=1.5):
    # Asymmetric loss for tails
    recon_loss = torch.mean(torch.abs(recon_x - x) ** alpha)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon_loss + kld

def train_vae(data, latent_dim=20, epochs=200, batch_size=128):
    scaler = RobustScaler()
    data_scaled = scaler.fit_transform(data)
    tensor_data = torch.tensor(data_scaled, dtype=torch.float32)
    dataset = TensorDataset(tensor_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    vae = ProductionVAE(latent_dim=latent_dim)
    optimizer = optim.Adam(vae.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)

    vae.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            x = batch[0]
            optimizer.zero_grad()
            recon_x, mu, logvar = vae(x)
            loss = asymmetric_vae_loss(recon_x, x, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        if (epoch + 1) % 50 == 0:
            logger.info(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

    return vae, scaler

def generate_samples(vae, scaler, n_samples=10000):
    vae.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, 20)
        samples = vae.decode(z).numpy()
    return scaler.inverse_transform(samples)

def calculate_var(returns, confidence=0.05):
    return np.percentile(returns, confidence * 100)

def backtest_var(real_returns, synthetic_samples, confidence=0.05):
    real_var = calculate_var(real_returns, confidence)
    synthetic_var = calculate_var(synthetic_samples, confidence)
    return real_var, synthetic_var

def kupiec_test(real_returns, synthetic_samples, confidence=0.05):
    # Simplified Kupiec test for VaR violations
    real_var = calculate_var(real_returns, confidence)
    synthetic_var = calculate_var(synthetic_samples, confidence)
    violations_real = np.sum(real_returns < real_var)
    violations_synth = np.sum(synthetic_samples < synthetic_var)
    expected_violations = len(real_returns) * (1 - confidence)
    # Chi-square test
    if violations_real == 0:
        return float('inf'), 0.0  # No violations, perfect
    chi2 = ((violations_real - expected_violations)**2) / expected_violations
    p_value = 1 - stats.chi2.cdf(chi2, 1)
    return chi2, p_value

def load_data(ticker, start="2010-01-01", end="2023-12-31"):
    try:
        data = yf.download(ticker, start=start, end=end, progress=False)
        if data is None or data.empty:
            raise ValueError(f"No data for {ticker}")
        price_col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
        returns = np.array(data[price_col].pct_change().dropna()).reshape(-1, 1)
        logger.info(f"Loaded {len(returns)} returns for {ticker}")
        return returns
    except Exception as e:
        logger.error(f"Error loading data for {ticker}: {e}")
        raise

def main():
    tickers = ['AAPL', 'MSFT', 'TSLA', 'SPY']
    results = {}

    for ticker in tickers:
        logger.info(f"Processing {ticker}...")
        try:
            returns = load_data(ticker)
            vae, scaler = train_vae(returns, epochs=200)
            samples = generate_samples(vae, scaler)

            real_var, synth_var = backtest_var(returns, samples)
            chi2, p_val = kupiec_test(returns, samples)
            ks_stat, ks_p = kstest(returns.flatten(), samples.flatten())

            results[ticker] = {
                'real_var': real_var,
                'synth_var': synth_var,
                'kupiec_chi2': chi2,
                'kupiec_p': p_val,
                'ks_stat': ks_stat,
                'ks_p': ks_p
            }
            logger.info(f"{ticker} - Real VaR: {real_var:.4f}, Synth VaR: {synth_var:.4f}, KS p: {ks_p:.4f}")
        except Exception as e:
            logger.error(f"Failed to process {ticker}: {e}")

    # Summary
    logger.info("Production VAE Results Summary:")
    for ticker, res in results.items():
        logger.info(f"{ticker}: VaR Real {res['real_var']:.4f} vs Synth {res['synth_var']:.4f} (Diff: {abs(res['real_var'] - res['synth_var']):.4f}), KS p={res['ks_p']:.4f}")

if __name__ == "__main__":
    main()
