# distribution_fitting.py
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from scipy import stats
from scipy.stats import gaussian_kde

sns.set(style="darkgrid")
os.makedirs("outputs", exist_ok=True)

# List of scipy.stats distribution names to try
DIST_NAMES = [
    "norm", "t", "laplace", "cauchy", "logistic", "gumbel_r", "genextreme"
    # note: lognorm, gamma, beta require positive support; we try but will catch errors
]

def download_returns(symbol, start, end):
    print(f"Downloading {symbol} from {start} to {end} ...")
    df = yf.download(symbol, start=start, end=end, progress=False)
    if df.empty:
        raise RuntimeError("No data returned. Check symbol, dates, or connectivity.")

    # Debug: print column names to see what's available
    print("Available columns:", df.columns.tolist())

    # Handle different column naming conventions in yfinance
    if 'Adj Close' in df.columns:
        price_col = 'Adj Close'
    elif 'Close' in df.columns:
        price_col = 'Close'
    else:
        # If it's a MultiIndex, try to get the close price
        price_col = df.columns[0] if len(df.columns) > 0 else 'Close'

    returns = df[price_col].pct_change().dropna()
    return returns

def fit_distribution_scipy(data, dist_name):
    """Fit one distribution via scipy, return dict of metrics."""
    dist = getattr(stats, dist_name)
    try:
        params = dist.fit(data)  # shape params (if any), loc, scale
        # Compute log-likelihood; clamp extreme zeros
        logpdf_vals = dist.logpdf(data, *params)
        # handle -inf
        finite_mask = np.isfinite(logpdf_vals)
        if not finite_mask.all():
            # if many -inf, penalize by setting large negative logpdf; keep finite ones
            logpdf_vals = np.where(finite_mask, logpdf_vals, -1e8)
        loglik = np.sum(logpdf_vals)
        k = len(params)  # number of fitted params (shape(s)+loc+scale)
        n = len(data)
        aic = 2*k - 2*loglik
        bic = k*np.log(n) - 2*loglik

        # KS test
        try:
            cdf_func = lambda x: dist.cdf(x, *params)
            ks_stat, ks_p = stats.kstest(data, cdf_func)
        except Exception:
            ks_stat, ks_p = np.nan, np.nan

        # KL divergence estimate using KDE vs fitted pdf over grid
        xs = np.linspace(np.min(data)-1e-6, np.max(data)+1e-6, 2000)
        kde = gaussian_kde(data)
        p_emp = kde(xs)
        p_emp = p_emp / np.trapz(p_emp, xs)
        p_fit = dist.pdf(xs, *params)
        # numerical fixes
        p_fit = np.where(p_fit <= 0, 1e-12, p_fit)
        p_fit = p_fit / np.trapz(p_fit, xs)
        kl = np.trapz(p_emp * np.log((p_emp + 1e-12) / (p_fit + 1e-12)), xs)

        return {
            "dist": dist_name,
            "params": params,
            "loglik": float(loglik),
            "k": k,
            "aic": float(aic),
            "bic": float(bic),
            "ks_stat": float(ks_stat),
            "ks_p": float(ks_p),
            "kl": float(kl)
        }
    except Exception as e:
        print(f"Failed fitting {dist_name}: {e}")
        return {
            "dist": dist_name,
            "params": None,
            "loglik": np.nan,
            "k": np.nan,
            "aic": np.nan,
            "bic": np.nan,
            "ks_stat": np.nan,
            "ks_p": np.nan,
            "kl": np.nan
        }

def fit_many(data, dist_names=DIST_NAMES):
    results = []
    for name in dist_names:
        print("Fitting", name)
        res = fit_distribution_scipy(data, name)
        results.append(res)
    return pd.DataFrame(results).sort_values(by="aic")

def plot_top_fits(data, results_df, top_n=3, symbol="DATA", out_prefix="outputs/fit"):
    # Plot histogram + KDE + top N fitted PDFs
    xs = np.linspace(np.min(data)*1.1, np.max(data)*1.1, 2000)
    plt.figure(figsize=(9,6))
    sns.histplot(data, bins=80, kde=True, stat='density', label='Empirical', alpha=0.6)
    colors = ['C0','C1','C2','C3','C4']
    top = results_df.dropna(subset=['params']).head(top_n)
    for i, row in enumerate(top.itertuples()):
        dist = getattr(stats, row.dist)
        params = row.params
        try:
            pdf_vals = dist.pdf(xs, *params)
            # normalize to density
            pdf_vals = np.where(pdf_vals < 0, 0, pdf_vals)
            plt.plot(xs, pdf_vals, label=f"{row.dist} (AIC={row.aic:.1f}, KL={row.kl:.3f})", linewidth=2)
        except Exception as e:
            print("Cannot plot", row.dist, e)
    plt.title(f"Empirical vs Top {top_n} fitted PDFs : {symbol}")
    plt.legend()
    out = f"{out_prefix}_{symbol}.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved plot:", out)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="AAPL", help="Ticker symbol to download")
    parser.add_argument("--start", default="2023-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2023-12-31", help="End date YYYY-MM-DD")
    parser.add_argument("--top", type=int, default=3, help="Number of top fits to plot")
    args = parser.parse_args()

    returns = download_returns(args.symbol, args.start, args.end)
    data = returns.values.flatten()  # numpy array, ensure 1D

    # quick summary
    print("N:", len(data), "mean:", np.mean(data), "std:", np.std(data))
    # Save raw returns
    pd.Series(data, name="returns").to_csv(f"outputs/{args.symbol}_returns.csv", index=False)

    # Fit distributions
    df_results = fit_many(data)
    df_results.to_csv(f"outputs/{args.symbol}_fit_results.csv", index=False)
    print("Saved results CSV: outputs/{}_fit_results.csv".format(args.symbol))

    # Print top 5 by AIC and by KL
    print("\nTop by AIC")
    print(df_results.nsmallest(5, "aic")[["dist","aic","kl","ks_stat","ks_p"]])
    print("\nTop by KL (smallest)")
    print(df_results.nsmallest(5, "kl")[["dist","aic","kl","ks_stat","ks_p"]])

    # Plot top fits
    plot_top_fits(data, df_results, top_n=args.top, symbol=args.symbol, out_prefix="outputs/fit")

if __name__ == "__main__":
    main()

