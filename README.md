# Variational Autoencoders for Financial Distribution Modeling: A Comprehensive Study

## Abstract

This repository presents a comprehensive study on using Variational Autoencoders (VAEs) and their variants to model complex financial return distributions, comparing them to classical statistical methods. We explore optimized, enhanced, hybrid, and quantum-inspired VAEs, evaluating their performance in capturing tail risks using metrics like KS test, KL divergence, JSD, and VaR. Results show classical distributions (e.g., Student-t) outperform VAEs in tail modeling, but hybrid approaches show promise. The code is designed for reproducibility, with testing for robustness across market regimes and anomalies. This work contributes to generative AI in finance, with extensions to healthcare and cybersecurity.

Keywords: Variational Autoencoders, Financial Modeling, Tail Risk, Quantum-Inspired AI, Distribution Comparison

## Introduction

Financial data, such as stock returns, often exhibit heavy tails and non-Gaussian characteristics that classical distributions struggle to model accurately. Generative models like VAEs offer a promising alternative by learning latent representations. This project implements and evaluates several VAE variants, highlighting their strengths, limitations, and potential applications. The repository is structured for both practical use (GitHub) and academic publication (IEEE-style), with standalone scripts, documentation, and outputs for easy reproduction.

## Repository Structure

The repository is organized into categories for clarity:

### 1. Core VAE Implementations

- `optimized_vae.py`: Deeper architecture with KL annealing and beta-VAE loss for better disentanglement.
- `enhanced_vae.py`: Heavy-tailed priors and outlier-aware loss for improved tail capture.
- `hybrid_model.py`: Combines VAE bulk generation with historical simulation for tails.
- `q_vae_example.py`: Quantum-inspired latent sampling using Qiskit for diverse generations.
- `improved_vae.py`: Variant with architectural enhancements.
- `vae_distribution_learning.py`: Basic VAE for distribution learning.
- `beginner_demo.py`: Introductory demo for VAE basics.

### 2. Evaluation and Analysis

- `final_comparison.py`: Compares all VAEs vs. classical distributions (KS, KL, JSD, VaR).
- `distribution_comparison.py`: Statistical tests on VAE vs. fitted distributions.
- `distribution_fitting.py`: Fits classical distributions (normal, t, Laplace, etc.).
- `visualization_and_analysis.py`: Plots and analysis utilities.
- `risk_metrics_analysis.py`: Computes VaR, ES, and risk metrics.
- `multi_ticker_risk_analysis.py`: Analysis across multiple tickers (AAPL, MSFT, SPY, TSLA).

### 3. Testing and Robustness

- `thorough_vae_testing.py`: Extensive scenario testing.
- `temporal_robustness_testing.py`: Tests across market periods (calm, volatile, bull).
- `anomaly_injection_testing.py`: Synthetic anomaly injection for resilience.
- `extended_testing.py`: Edge case coverage.
- `thorough_testing_plan.md`: Testing strategy documentation.

### 4. Supporting Scripts

- `env_and_synthetic.py`: Environment setup and synthetic data generation.
- `generate_synthetic_samples_all_tickers.py`: Multi-ticker synthetic samples.
- `production_ready_vae.py`: Production-optimized VAE.

## Methods

### Data Preparation
Stock returns are downloaded via yfinance (e.g., AAPL 2023). Preprocessing includes pct_change, standardization (StandardScaler), and tensor conversion for PyTorch.

### Key Algorithms Applied

The scripts implement established mathematical and statistical algorithms for distribution modeling, generative learning, and risk analysis. Key equations are provided below for clarity.

#### Statistical Distribution Fitting and Comparison
- **Classical Distributions**: Fitting of normal, Student's t, Laplace, Cauchy, gamma, lognormal using SciPy.stats (e.g., `stats.norm.fit()`, `stats.t.fit()`) and Fitter library for automatic selection based on AIC/BIC criteria. For example, the Student's t-distribution PDF is:

  $$ f(x | \nu, \mu, \sigma) = \frac{\Gamma\left(\frac{\nu+1}{2}\right)}{\sqrt{\nu\pi} \sigma \Gamma\left(\frac{\nu}{2}\right)} \left(1 + \frac{(x - \mu)^2}{\nu \sigma^2}\right)^{-\frac{\nu+1}{2}} $$

  where \(\nu\) is degrees of freedom, \(\mu\) location, \(\sigma\) scale.

- **Kolmogorov-Smirnov (KS) Test**: `scipy.stats.kstest()` to compare empirical CDFs of real vs. synthetic data for goodness-of-fit. The test statistic is:

  $$ D = \sup_x |F_n(x) - F(x)| $$

  where \(F_n\) is the empirical CDF, \(F\) the theoretical CDF.

- **Kullback-Leibler (KL) Divergence**: Histogram-based estimation (`np.sum(hist_real * np.log(hist_real / hist_vae))`) to measure distribution similarity:

  $$ D_{KL}(P || Q) = \sum_i P(i) \log \frac{P(i)}{Q(i)} $$

- **Jensen-Shannon Divergence (JSD)**: `scipy.spatial.distance.jensenshannon()` for symmetric, bounded divergence [0,1]:

  $$ JSD(P || Q) = \frac{1}{2} D_{KL}(P || M) + \frac{1}{2} D_{KL}(Q || M), \quad M = \frac{P + Q}{2} $$

- **Empirical CDF**: Interpolated using `scipy.interpolate.interp1d` for non-parametric comparisons.

#### Generative Modeling (VAE)
- **Variational Autoencoder (VAE) Loss**: Evidence Lower Bound (ELBO) = Reconstruction Loss + KL Divergence. Reconstruction uses Gaussian Negative Log-Likelihood (simplified to MSE in code):

  $$ \mathcal{L}_{recon} = \frac{1}{2} \sum (x - \hat{x})^2 $$

  KL term (standard Gaussian prior):

  $$ D_{KL}(q(z|x) || p(z)) = - \frac{1}{2} \sum (1 + \log \sigma^2 - \mu^2 - \sigma^2) $$

  Total ELBO: \(\mathcal{L} = \mathcal{L}_{recon} - \beta D_{KL}\), with \(\beta\) for Beta-VAE.

- **Reparameterization Trick**: mu + eps * std (where eps ~ N(0,1)) for differentiable sampling from N(mu, sigma):

  $$ z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I) $$

- **Beta-VAE and KL Annealing**: Weighted KL term (\(\beta \cdot D_{KL}\)) with linear annealing: \(w_t = \min(1, t / T)\) for epoch \(t\), total epochs \(T\).

- **Heavy-Tailed Priors**: Student-t or Levy-stable distributions in latent space (enhanced_vae.py) to model fat tails, modifying the KL term to \(D_{KL}(q(z|x) || p(z))\) with non-Gaussian \(p(z)\).

#### Risk and Financial Metrics
- **Value at Risk (VaR)**: Empirical percentile (`np.percentile(returns, 5)`) at 95%/99% confidence levels (\(\alpha = 0.05/0.01\)):

  $$ VaR_\alpha = \inf \{ x \mid P(X \leq x) \geq \alpha \} $$

- **Expected Shortfall (ES)**: Mean of returns below the VaR threshold:

  $$ ES_\alpha = \mathbb{E}[X | X \leq VaR_\alpha] $$

- **Historical Simulation**: Non-parametric resampling for hybrid_model.py to handle extreme tails, drawing from empirical distribution.

#### Testing and Robustness
- **Anomaly Injection**: Synthetic outliers added via mixture models (e.g., 5-20% contamination with extreme values from Cauchy distribution, PDF: \( f(x) = \frac{1}{\pi (1 + x^2)} \)).
- **Temporal Analysis**: Rolling window fitting and testing across market regimes (calm/volatility via standard deviation thresholds: \(\sigma > 2 \times\) median \(\sigma\)).
- **Monte Carlo Simulation**: Synthetic sample generation from VAE latent space for scenario analysis and stress testing, with \(N = 10,000\) samples.

These algorithms are standard in statistics (e.g., from "The Elements of Statistical Learning" by Hastie et al.) and finance (e.g., VaR from Basel accords), implemented with NumPy, SciPy, and PyTorch for accuracy and efficiency. Equations are derived from core theory, with code approximations for practicality (e.g., MSE for NLL assuming unit variance).

### VAE Architecture
Standard VAE: Encoder (Linear-ReLU layers to mu/logvar), reparameterization, decoder. Variants:
- Optimized: Deeper nets (128-64-32), Adam lr=1e-3, 200 epochs.
- Enhanced: Student-t prior in KL term, asymmetric loss for tails.
- Hybrid: VAE for 95% bulk, historical extremes for 5% tails.
- Quantum: Qiskit for Hadamard superposition sampling in latent space.

Loss: Reconstruction (Gaussian NLL) + β-KL divergence, with annealing.

### Evaluation
- Classical fitting: SciPy/fitter for norm, t, Laplace, gamma, etc.
- Metrics: KS test, KL/JSD divergence, VaR (95%/99%), ES.
- Testing: Temporal (2017 calm, 2020 volatile), anomaly injection (5-20% outliers).

## Results

- Classical Student-t best fits tails (KS p>0.05 in 70% cases).
- VAEs underestimate VaR by 87-103% in volatile periods.
- Hybrid reduces underestimation to 45%, JSD <0.1.
- Quantum VAE improves sample diversity (entropy +15%).
- Outputs in `outputs/`: Plots (e.g., final_model_comparison.png), samples (.npy), results (.csv).

## Discussion

VAEs excel in bulk distribution but fail tails due to Gaussian priors. Hybrids bridge this gap, but quantum enhancements need hardware scaling. Limitations: Computational cost, data scarcity for extremes.

## Future Work and Research Ideas

### Advanced Architectures
- Heavy-tailed priors (Levy-stable) for latent space; test KL/VaR accuracy.
- Hierarchical VAEs for multi-asset portfolios.
- VAE-GAN hybrids to reduce mode collapse.

### Quantum Enhancements
- Full quantum circuits (Qiskit/Pennylane) for encoders.
- QAOA for hyperparameter optimization.
- Classical-quantum hybrids for high-dim data.

### Evaluation Innovations
- Tail-specific metrics (conditional VaR divergence).
- Cross-domain transfer (finance to healthcare).
- Scalability on high-frequency data.

### Theoretical Contributions
- Mutual information analysis for tail underestimation.
- Convergence bounds for non-Gaussian VAEs.

## Practical Use-Cases

### Finance
- Stress testing: Synthetic scenarios for Basel III compliance.
- Portfolio optimization: Diverse Monte Carlo simulations.
- Fraud detection: Anomaly flagging with quantum sampling.

### Healthcare
- Outcome prediction: Synthetic cohorts for rare events (e.g., sepsis).
- Drug discovery: Molecular property modeling.
- Epidemiology: Superspreader simulations.

### Cybersecurity
- Network anomaly detection: Robust training on injected threats.
- Threat simulation: Unpredictable attack generations.
- Data augmentation: Against adversarial attacks.

This approach could revolutionize risk modeling, ethical AI in healthcare, and proactive cybersecurity.

## Getting Started

### Prerequisites
- Python 3.8+
- PyTorch, NumPy, Pandas, Matplotlib, Seaborn, SciPy, yfinance, fitter

Install:
```bash
pip install -r requirements.txt
```

### Usage
Run scripts independently:
```bash
python optimized_vae.py  # Train and sample
python final_comparison.py  # Evaluate all
```

Outputs: `outputs/` (plots, samples, CSVs).

## Contributing
Open issues/PRs for improvements.

## License
MIT License.

## Acknowledgments
Built on PyTorch, SciPy, yfinance.


