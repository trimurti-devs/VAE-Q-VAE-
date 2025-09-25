# improved_vae.py
# Further improved VAE with advanced architecture, cyclic KL annealing, and early stopping

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# 1. Load Dataset
# =========================
print("Downloading stock data...")
data = yf.download("AAPL", start="2018-01-01", end="2018-12-31")

if data is None or data.empty:
    raise RuntimeError("Failed to download data or data is empty")

if 'Adj Close' in data.columns:
    price_col = 'Adj Close'
elif 'Close' in data.columns:
    price_col = 'Close'
else:
    price_col = data.columns[0] if len(data.columns) > 0 else 'Close'

returns = np.array(data[price_col].pct_change().dropna()).reshape(-1, 1)

scaler = StandardScaler()
returns_scaled = scaler.fit_transform(returns)

tensor_data = torch.tensor(returns_scaled, dtype=torch.float32)
dataset = TensorDataset(tensor_data)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# =========================
# 2. Define Improved VAE Model
# =========================
class ImprovedVAE(nn.Module):
    def __init__(self, input_dim=1, latent_dim=10):
        super(ImprovedVAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder with BatchNorm and Dropout
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

        # Decoder symmetric to encoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
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

# =========================
# 3. Cyclic KL Annealing Scheduler
# =========================
def cyclic_kl_annealing(epoch, cycle_length=50, max_beta=1.0):
    cycle_pos = epoch % cycle_length
    beta = max_beta * (cycle_pos / cycle_length)
    return beta

# =========================
# 4. Loss Function
# =========================
def vae_loss(recon_x, x, mu, logvar, beta):
    recon_loss = nn.MSELoss(reduction='sum')(recon_x, x)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kld

# =========================
# 5. Training Loop with Early Stopping
# =========================
vae = ImprovedVAE(input_dim=1, latent_dim=10)
optimizer = optim.Adam(vae.parameters(), lr=1e-3)

epochs = 200
early_stop_patience = 20
best_loss = float('inf')
patience_counter = 0

vae.train()
for epoch in range(epochs):
    total_loss = 0
    beta = cyclic_kl_annealing(epoch, cycle_length=50, max_beta=1.0)
    for batch in dataloader:
        x = batch[0]
        optimizer.zero_grad()
        recon_x, mu, logvar = vae(x)
        loss = vae_loss(recon_x, x, mu, logvar, beta)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader.dataset)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Beta: {beta:.3f}")

    # Early stopping check
    if avg_loss < best_loss:
        best_loss = avg_loss
        patience_counter = 0
        torch.save(vae.state_dict(), "outputs/improved_vae.pth")
    else:
        patience_counter += 1
        if patience_counter >= early_stop_patience:
            print("Early stopping triggered.")
            break

# =========================
# 6. Sampling and Plotting
# =========================
vae.eval()
with torch.no_grad():
    z = torch.randn(10000, vae.latent_dim)
    samples = vae.decode(z).numpy()

samples = scaler.inverse_transform(samples)
np.save("outputs/improved_vae_samples.npy", samples)
print("✅ Improved VAE samples saved as improved_vae_samples.npy")

plt.figure(figsize=(10, 6))
sns.histplot(returns, bins=50, color="blue", label="Real Data", kde=True, stat="density")
sns.histplot(samples, bins=50, color="purple", label="Improved VAE Samples", kde=True, stat="density", alpha=0.6)
plt.legend()
plt.title("Real vs Improved VAE Learned Distribution (AAPL Returns)")
plt.show()
