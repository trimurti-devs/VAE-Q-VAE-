# env_and_synthetic.py
import sys, platform
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import sklearn

print("Python:", sys.version.splitlines()[0])
print("Platform:", platform.platform())
print("numpy:", np.__version__)
print("pandas:", pd.__version__)
print("matplotlib:", matplotlib.__version__)
print("seaborn:", sns.__version__)
print("scipy:", scipy.__version__)
print("sklearn:", sklearn.__version__)

# Create output folder
import os
os.makedirs("outputs", exist_ok=True)

# Generate synthetic datasets
rng = np.random.default_rng(42)

n = 5000
data_normal = rng.normal(loc=0, scale=1, size=n)
data_lognorm = rng.lognormal(mean=0.0, sigma=0.6, size=n)
data_mixture = np.concatenate([rng.normal(-2, 0.5, size=int(n*0.4)), rng.normal(1.5, 0.8, size=int(n*0.6))])

datasets = {
    "normal": data_normal,
    "lognormal": data_lognorm,
    "mixture": data_mixture
}

for name, arr in datasets.items():
    plt.figure(figsize=(7,4))
    sns.histplot(arr, bins=80, kde=True)
    plt.title(f"Histogram + KDE : {name}")
    out = f"outputs/synthetic_{name}.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved:", out)

print("Synthetic sample means/stds:")
for name, arr in datasets.items():
    print(name, "mean:", np.mean(arr), " std:", np.std(arr))
