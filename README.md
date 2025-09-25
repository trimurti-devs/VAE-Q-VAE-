Financial Distribution Modeling with Variational Autoencoders (VAEs)

This repository contains a comprehensive research project exploring the use of Variational Autoencoders (VAEs), including optimized, enhanced, hybrid, and quantum-inspired variants, for modeling complex financial return distributions. The goal is to evaluate their effectiveness compared to classical statistical distributions, especially in capturing tail risk.

---

Repository Contents

The repository is organized into the following categories for clarity and ease of use:

### 1. Core VAE Implementations

These scripts contain the main VAE models developed and tested in this project:

- `optimized_vae.py`: Production-ready optimized VAE with deeper architecture, KL annealing, and beta-VAE loss.
- `enhanced_vae.py`: VAE with heavy-tailed latent priors and outlier-aware loss functions.
- `hybrid_model.py`: Hybrid approach combining VAE-generated bulk samples with historical tail simulation.
- `q_vae_example.py`: Quantum-inspired VAE demonstrating latent space sampling using quantum randomness.
- `improved_vae.py`: Additional VAE variant with architectural and training improvements.
- `vae_distribution_learning.py`: Basic VAE implementation for learning distributions.
- `beginner_demo.py`: Simple VAE demo for newcomers to understand the basics.

### 2. Evaluation and Analysis

Scripts for evaluating model performance, fitting classical distributions, and analyzing results:

- `final_comparison.py`: Comprehensive evaluation comparing all VAE variants and classical distributions.
- `distribution_comparison.py`: Comparison of VAE samples with classical fitted distributions using statistical tests.
- `distribution_fitting.py`: Classical distribution fitting on financial data.
- `visualization_and_analysis.py`: Visualization utilities and analysis scripts.
- `risk_metrics_analysis.py`: Calculation and analysis of financial risk metrics.
- `multi_ticker_risk_analysis.py`: Risk analysis across multiple financial tickers.

### 3. Testing and Robustness

Scripts designed to test model robustness, temporal stability, and anomaly handling:

- `thorough_vae_testing.py`: Extensive testing of VAE models across scenarios.
- `temporal_robustness_testing.py`: Testing model performance across calm, volatile, and bull market periods.
- `anomaly_injection_testing.py`: Injecting synthetic anomalies to test model resilience.
- `extended_testing.py`: Additional testing scripts covering edge cases.
- `thorough_testing_plan.md`: Documentation of the testing strategy and coverage.

### 4. Supporting and Utility Scripts

Additional scripts for data preparation, synthetic data generation, and production readiness:

- `env_and_synthetic.py`: Environment setup and synthetic dataset generation.
- `generate_synthetic_samples_all_tickers.py`: Script to generate synthetic samples for multiple tickers.
- `production_ready_vae.py`: VAE implementation optimized for production deployment.

---

Getting Started

Prerequisites

- Python 3.8+
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Seaborn
- SciPy
- yfinance
- fitter

Install dependencies via:

pip install -r requirements.txt

Running the Models

Each script is standalone and can be run independently. For example:

python optimized_vae.py
python q_vae_example.py
python hybrid_model.py
python enhanced_vae.py
python final_comparison.py

Outputs

- Model checkpoints and generated samples are saved in the outputs/ directory.
- Evaluation plots and comparison charts are saved as PNG files in outputs/.

---

Research Highlights

- Classical Student-t distribution outperforms all VAE variants in tail risk modeling.
- VAEs tend to underestimate Value at Risk (VaR) significantly.
- Hybrid models combining VAE bulk generation with historical tail simulation show promise but still lag classical methods.
- Quantum-inspired VAEs are an emerging area with potential for future breakthroughs.
- Detailed evaluation metrics and plots are provided for all models.

---

How to Use This Repository

- Explore different VAE architectures and their training scripts.
- Use final_comparison.py to reproduce the evaluation and comparison results.
- Review research_ideas_and_usecases.md for inspiration on future research directions.
- Extend or modify models for your own financial data or other domains.

---

Contributing

Contributions are welcome! Please open issues or pull requests for bug fixes, improvements, or new features.

---

License

This project is licensed under the MIT License.

---

Contact

For questions or collaboration, please contact the author.
