# extended_testing.py
# Script to perform extended testing for robustness and real-world applicability

import yfinance as yf
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from vae_distribution_learning import VAE
from optimized_vae import OptimizedVAE
from q_vae_example import QVAE, quantum_random_samples
from scipy import stats
from scipy.interpolate import interp1d
from scipy.spatial.distance import jensenshannon

# List of tickers for dataset diversity
tickers = ["AAPL", "MSFT", "GOOG", "TSLA", "SPY"]

# Time periods for temporal robustness
periods = [
    ("2018-01-01", "2018-12-31"),  # Calm market
    ("2020-02-01", "2020-04-30"),  # COVID crash
    ("2022-01-01", "2022-12-31"),  # Recent year
]

def load_returns(ticker, start, end):
    data = yf.download(ticker, start=start, end=end, progress=False)
    if data is None or data.empty:
        raise RuntimeError(f"No data for {ticker} {start} to {end}")
    if 'Adj Close' in data.columns:
        price_col = 'Adj Close'
    elif 'Close' in data.columns:
        price_col = 'Close'
    else:
        price_col = data.columns[0]
    returns = np.array(data[price_col].pct_change().dropna()).reshape(-1, 1)
    return returns

def prepare_dataloader(returns):
    scaler = StandardScaler()
    returns_scaled = scaler.fit_transform(returns)
    tensor_data = torch.tensor(returns_scaled, dtype=torch.float32)
    dataset = TensorDataset(tensor_data)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    return dataloader, scaler

def train_and_evaluate_vae(dataloader, scaler, model_class, epochs=50, quantum=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_class().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Determine latent dimension dynamically
    latent_dim = None
    if hasattr(model, 'latent_dim'):
        latent_dim = model.latent_dim
    elif hasattr(model, 'fc_mu'):
        latent_dim = model.fc_mu.out_features
    else:
        latent_dim = 2  # default fallback

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            recon_x, mu, logvar = model(x)
            recon_loss = 0.5 * torch.sum((recon_x - x)**2) / x.size(0)
            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
            loss = recon_loss + kld
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

    model.eval()
    with torch.no_grad():
        if quantum:
            z_samples = quantum_random_samples(5000, latent_dim)
            # Ensure z_samples shape matches model latent dim
            if z_samples.shape[1] != latent_dim:
                raise ValueError(f"Quantum samples latent dim {z_samples.shape[1]} does not match model latent dim {latent_dim}")
            z = torch.tensor(z_samples, dtype=torch.float32).to(device)
        else:
            z = torch.randn(5000, latent_dim).to(device)
        samples = model.decode(z).cpu().numpy()
    samples = scaler.inverse_transform(samples)
    return samples

def empirical_cdf(data):
    data_sorted = np.sort(data)
    n = len(data)
    y = np.arange(1, n+1) / n
    return interp1d(data_sorted, y, kind='linear', bounds_error=False, fill_value=0.0)

def evaluate_distribution(real, generated):
    ks = stats.kstest(real, empirical_cdf(generated.flatten()))
    hist_real, bin_edges = np.histogram(real, bins=100, density=True)
    hist_gen, _ = np.histogram(generated, bins=bin_edges, density=True)
    hist_real += 1e-8
    hist_gen += 1e-8
    hist_real /= hist_real.sum()
    hist_gen /= hist_gen.sum()
    kl_div = np.sum(hist_real * np.log(hist_real / hist_gen))
    jsd = jensenshannon(hist_real, hist_gen)**2
    return ks, kl_div, jsd

def main():
    results = []
    for ticker in tickers:
        for start, end in periods:
            print(f"\nTesting {ticker} from {start} to {end}")
            returns = load_returns(ticker, start, end)
            dataloader, scaler = prepare_dataloader(returns)

            # Original VAE
            print("Training Original VAE...")
            orig_samples = train_and_evaluate_vae(dataloader, scaler, VAE, epochs=50)
            ks_o, kl_o, jsd_o = evaluate_distribution(returns.flatten(), orig_samples)

            # Optimized VAE
            print("Training Optimized VAE...")
            opt_samples = train_and_evaluate_vae(dataloader, scaler, OptimizedVAE, epochs=100)
            ks_opt, kl_opt, jsd_opt = evaluate_distribution(returns.flatten(), opt_samples)

            # Q-VAE
            print("Training Q-VAE...")
            q_samples = train_and_evaluate_vae(dataloader, scaler, QVAE, epochs=100, quantum=True)
            ks_q, kl_q, jsd_q = evaluate_distribution(returns.flatten(), q_samples)

            results.append({
                "ticker": ticker,
                "period": f"{start} to {end}",
                "ks_original": ks_o.pvalue,
                "kl_original": kl_o,
                "jsd_original": jsd_o,
                "ks_optimized": ks_opt.pvalue,
                "kl_optimized": kl_opt,
                "jsd_optimized": jsd_opt,
                "ks_qvae": ks_q.pvalue,
                "kl_qvae": kl_q,
                "jsd_qvae": jsd_q,
            })

    # Summarize results
    import pandas as pd
    df = pd.DataFrame(results)
    print("\nSummary of Testing Results:")
    print(df)
    df.to_csv("outputs/extended_testing_results.csv", index=False)
    print("Saved extended testing results to outputs/extended_testing_results.csv")

if __name__ == "__main__":
    main()
