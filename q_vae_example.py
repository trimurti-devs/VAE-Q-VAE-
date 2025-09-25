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

# Try to import Qiskit
qiskit_available = False
try:
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
    qiskit_available = True
    print("Qiskit available, using quantum random sampling.")
except Exception as e:
    print(f"Qiskit not available or failed to import ({e}). Using classical random with quantum-inspired simulation.")
    qiskit_available = False

# =========================
# Quantum-Inspired Random Sampling
# =========================
def quantum_random_samples(n_samples, latent_dim, n_qubits=5):
    """
    Generate quantum-inspired random samples using Qiskit.
    Uses a simple quantum circuit to generate random bits, then map to Gaussian-like.
    """
    if not qiskit_available:
        # Fallback to classical random
        return np.random.randn(n_samples, latent_dim)

    simulator = AerSimulator()
    samples = []

    for _ in range(n_samples):
        qc = QuantumCircuit(n_qubits)
        # Add Hadamard gates for superposition
        for i in range(n_qubits):
            qc.h(i)
        # Measure
        qc.measure_all()

        # Transpile and run
        transpiled = transpile(qc, simulator)
        job = simulator.run(transpiled, shots=1)
        result = job.result()
        counts = result.get_counts()

        # Convert bitstring to float
        bitstring = list(counts.keys())[0]
        # Map to latent dim by repeating or truncating
        bits = [int(b) for b in bitstring]
        while len(bits) < latent_dim:
            bits += bits
        bits = bits[:latent_dim]
        # Convert to Gaussian-like: 0-> -1, 1->1, but randomize
        sample = np.array([2 * b - 1 + np.random.normal(0, 0.5) for b in bits])
        samples.append(sample)

    return np.array(samples)

class QVAE(nn.Module):
    def __init__(self, input_dim=1, latent_dim=10):
        super(QVAE, self).__init__()

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
        eps = torch.randn_like(std)  # Classical reparameterization
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

    qvae = QVAE(input_dim=1, latent_dim=10)
    optimizer = optim.Adam(qvae.parameters(), lr=1e-3)

    epochs = 100
    qvae.train()
    print(f"Training Q-VAE for {args.symbol}...")
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            x = batch[0]
            optimizer.zero_grad()
            recon_x, mu, logvar = qvae(x)
            loss = vae_loss(recon_x, x, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

    qvae.eval()
    latent_dim = 10
    n_samples = 5000
    print("Generating quantum-inspired latent samples...")
    z_quantum = torch.tensor(quantum_random_samples(n_samples, latent_dim), dtype=torch.float32)
    with torch.no_grad():
        samples = qvae.decode(z_quantum).numpy()

    samples = scaler.inverse_transform(samples)
    os.makedirs("outputs", exist_ok=True)
    np.save(f"outputs/q_vae_samples_{args.symbol}.npy", samples)
    print(f"✅ Q-VAE samples saved as outputs/q_vae_samples_{args.symbol}.npy")

    # Plot Comparison
    plt.figure(figsize=(10, 6))
    sns.histplot(returns, bins=50, color="blue", label="Real Data", kde=True, stat="density")
    sns.histplot(samples, bins=50, color="green", label="Q-VAE Samples", kde=True, stat="density", alpha=0.6)
    plt.legend()
    plt.title(f"Real vs Q-VAE Learned Distribution ({args.symbol} Returns)")
    plt.show()

if __name__ == "__main__":
    main()
