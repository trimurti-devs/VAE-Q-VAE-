# Variational Autoencoders for Financial Distribution Modeling: A Comprehensive Study

## Abstract

This repository presents a comprehensive study on using **Variational Autoencoders (VAEs)** and their variants to model complex **financial return distributions**, comparing them with classical statistical approaches. We explore **optimized**, **enhanced**, **hybrid**, and **quantum-inspired** VAE architectures and evaluate their performance in capturing **tail risks** using metrics such as **KS test**, **KL divergence**, **JSD**, and **VaR**.

Results show that **classical distributions** (e.g., Student-t) outperform VAEs in tail modeling, but **hybrid methods** show promise. The repository is designed for **reproducibility**, including tests for robustness across **market regimes** and **anomalies**. This work contributes to **generative AI in finance**, with potential extensions to **healthcare** and **cybersecurity**.

**Keywords:** Variational Autoencoders, Financial Modeling, Tail Risk, Quantum-Inspired AI, Distribution Comparison

---

## Introduction

Financial data (e.g., stock returns) often exhibit **heavy tails** and **non-Gaussian behavior**, which classical distributions may fail to capture. Generative models such as **VAEs** can learn latent representations directly from data, offering a flexible alternative.

This project implements and evaluates multiple VAE variants, discusses their **strengths, limitations, and use cases**, and is structured for **academic publication** and **practical usage**.

---

## Repository Structure

├── Core VAE Implementations
│ ├── optimized_vae.py
│ ├── enhanced_vae.py
│ ├── hybrid_model.py
│ ├── q_vae_example.py
│ ├── improved_vae.py
│ └── beginner_demo.py
│
├── Evaluation and Analysis
│ ├── final_comparison.py
│ ├── distribution_comparison.py
│ ├── risk_metrics_analysis.py
│ └── multi_ticker_risk_analysis.py
│
├── Testing and Robustness
│ ├── thorough_vae_testing.py
│ ├── temporal_robustness_testing.py
│ └── anomaly_injection_testing.py
│
├── Supporting Scripts
│ ├── env_and_synthetic.py
│ └── generate_synthetic_samples_all_tickers.py
│
└── outputs/


---

## Methods

### Data Preparation

Stock returns are fetched using `yfinance` (e.g., AAPL 2023). Preprocessing includes:

- Percentage change (`pct_change`)
- Standardization (`StandardScaler`)
- Tensor conversion (PyTorch)

---

## Statistical Distribution Modeling

### Classical Distributions

Fitted using `scipy.stats` or `fitter`:

- Normal
- Student-t
- Laplace
- Cauchy
- Gamma
- Lognormal

Example: **Student-t distribution PDF**

$$
f(x \mid \nu, \mu, \sigma) = \frac{\Gamma\left(\frac{\nu+1}{2}\right)}{\sqrt{\nu \pi}\,\sigma\,\Gamma\left(\frac{\nu}{2}\right)} \left( 1 + \frac{(x - \mu)^2}{\nu \sigma^2} \right)^{-\frac{\nu+1}{2}}
$$

---

### Kolmogorov-Smirnov (KS) Test

Compares empirical CDF with theoretical:

$$
D = \sup_x \, |F_n(x) - F(x)|
$$

---

### Kullback–Leibler (KL) Divergence

$$
D_{\mathrm{KL}}(P \parallel Q) = \sum_i P(i) \log \frac{P(i)}{Q(i)}
$$

---

### Jensen–Shannon Divergence (JSD)

$$
\mathrm{JSD}(P \parallel Q) = \frac{1}{2} D_{\mathrm{KL}}(P \parallel M) + \frac{1}{2} D_{\mathrm{KL}}(Q \parallel M)
$$

where:

$$
M = \frac{P + Q}{2}
$$

---

## Variational Autoencoder (VAE)

### ELBO Objective

The Evidence Lower Bound (ELBO):

$$
\mathcal{L} = \mathcal{L}_{\text{recon}} - \beta D_{\mathrm{KL}}
$$

- **Reconstruction loss** (Gaussian NLL, often approximated as MSE):

$$
\mathcal{L}_{\text{recon}} = \frac{1}{2} \sum_{i=1}^N (x_i - \hat{x}_i)^2
$$

- **KL Divergence term** (for Gaussian prior \( p(z) = \mathcal{N}(0, I) \)):

$$
D_{\mathrm{KL}}(q(z|x) \parallel p(z)) = -\frac{1}{2} \sum_{j=1}^J \left[ 1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2 \right]
$$

---

### Reparameterization Trick

$$
z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

---

### Heavy-Tailed Priors

In enhanced VAEs, \( p(z) \) is replaced with **Student-t** or **Lévy-stable** distributions to capture tail behavior:

$$
D_{\mathrm{KL}}(q(z|x) \parallel p(z))
$$

---

## Financial Risk Metrics

### Value at Risk (VaR)

At confidence level \( \alpha \):

$$
\mathrm{VaR}_\alpha = \inf \{ x \mid P(X \le x) \ge \alpha \}
$$

---

### Expected Shortfall (ES)

$$
\mathrm{ES}_\alpha = \mathbb{E}[X \mid X \le \mathrm{VaR}_\alpha]
$$

---

### Historical Simulation

Used in hybrid models: Resample from empirical distribution for extreme tails.

---

## Results

- **Student-t** distribution best models tails (KS p > 0.05 in ~70% of cases).
- VAEs **underestimate VaR by 87–103%** in volatile periods.
- **Hybrid models** reduce VaR underestimation to ~45% (JSD < 0.1).
- **Quantum-inspired VAEs** increase latent diversity (+15% entropy).

---

## Discussion

- VAEs model **bulk distributions** well but struggle with **tails** due to Gaussian priors.
- Hybrid approaches offer a practical solution.
- Quantum methods require hardware scaling but show **diversity gains**.

---

## Future Directions

- **Heavy-tailed latent priors** (e.g., Lévy)  
- **Hierarchical VAEs** for multi-asset portfolios  
- **VAE-GAN hybrids** for reduced mode collapse  
- **Quantum encoders** and **QAOA optimization**

---

## Applications

### Finance
- Stress testing (Basel III)
- Portfolio optimization
- Fraud/anomaly detection

### Healthcare
- Rare event simulation (e.g., sepsis)
- Molecular generation
- Epidemiological modeling

### Cybersecurity
- Threat generation and simulation
- Anomaly-robust training
- Data augmentation

---

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch, NumPy, Pandas, Matplotlib, Seaborn, SciPy, yfinance, fitter

```bash
pip install -r requirements.txt



