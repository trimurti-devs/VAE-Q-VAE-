# generate_synthetic_samples_all_tickers.py
# Script to generate synthetic samples for all tickers using VAE, Optimized VAE, and Q-VAE

import subprocess

tickers = ["AAPL", "MSFT", "TSLA", "SPY"]

def run_script(script_name, ticker):
    print(f"Running {script_name} for {ticker}...")
    # Run the script with ticker argument, save samples with ticker suffix
    # Assumes scripts accept --symbol argument and save outputs accordingly
    cmd = f"D:\\python3.9\\python.exe {script_name} --symbol {ticker}"
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Error running {script_name} for {ticker}")
    else:
        print(f"Completed {script_name} for {ticker}")

if __name__ == "__main__":
    # Run vae_distribution_learning.py for each ticker
    for ticker in tickers:
        run_script("vae_distribution_learning.py", ticker)

    # Run optimized_vae.py for each ticker
    for ticker in tickers:
        run_script("optimized_vae.py", ticker)

    # Run q_vae_example.py for each ticker
    for ticker in tickers:
        run_script("q_vae_example.py", ticker)

    print("Synthetic sample generation completed for all tickers.")
