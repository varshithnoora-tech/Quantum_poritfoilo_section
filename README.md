# Quantum_poritfoilo_section[quantum_portfolio_selection.py](https://github.com/user-attachments/files/33497201/quantum_portfolio_selection.py)
# %% [markdown]
# # Quantum Portfolio Selection (QUBO)
#
# This notebook implements the idea from the proposal:
#
# 1. Turn stock selection into a **QUBO** (Quadratic Unconstrained Binary Optimization) problem.
# 2. Add budget / cardinality limits as **penalty terms**.
# 3. Solve a small universe (20 stocks) with a **classical benchmark** (simulated annealing)
#    and, optionally, with **QAOA** (Qiskit).
# 4. **Scale up** by clustering stocks by correlation, solving each cluster, and combining results.
#
# The file is written as a percent-format script (`# %%` cells), so it can be opened as a
# Jupyter notebook with Jupytext, or run directly with `python quantum_portfolio_selection.py`.

# %% [markdown]
# ## 1. Setup
#
# Required packages (install once):
#
# ```bash
# pip install numpy pandas yfinance scipy matplotlib
# pip install qiskit qiskit-algorithms qiskit-optimization   # optional, for QAOA
# ```
#
# If you don't install the quantum packages, the notebook still runs fully with the classical benchmark.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import time

rng = np.random.default_rng(42)

try:
    import yfinance as yf
    HAVE_YF = True
except ImportError:
    HAVE_YF = False

try:
    from qiskit_optimization import QuadraticProgram
    from qiskit_optimization.converters import QuadraticProgramToQubo
    from qiskit_algorithms import QAOA
    from qiskit_algorithms.optimizers import COBYLA
    from qiskit_algorithms.utils import algorithm_globals
    from qiskit.primitives import Sampler
    from qiskit_optimization.algorithms import MinimumEigenOptimizer
    HAVE_QISKIT = True
    algorithm_globals.random_seed = 42
except ImportError:
    HAVE_QISKIT = False

print("yfinance available:", HAVE_YF)
print("Qiskit available:  ", HAVE_QISKIT)

# %% [markdown]
# ## 2. Data
#
# We use real daily returns when `yfinance` is available and internet access works.
# Otherwise we generate synthetic returns with a realistic correlation structure, so the
# notebook always runs.

# %%
TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "JPM", "BAC", "WFC", "GS",
    "XOM", "CVX", "JNJ", "PFE", "UNH", "PG", "KO", "PEP", "WMT", "HD",
]

def load_returns(tickers, years=3):
    """Return a DataFrame of daily log returns (rows = days, columns = tickers)."""
    if HAVE_YF:
        try:
            data = yf.download(tickers, period=f"{years}y", auto_adjust=True, progress=False)["Close"]
            data = data.dropna(axis=1, how="any")
            return np.log(data / data.shift(1)).dropna()
        except Exception as exc:
            print("Download failed, using synthetic data:", exc)
    return synthetic_returns(len(tickers), tickers)

def synthetic_returns(n, names, days=750):
    """Synthetic correlated returns: a few sector factors plus stock-specific noise."""
    n_factors = 4
    loadings = rng.uniform(0.3, 1.0, size=(n, n_factors)) * (rng.random((n, n_factors)) < 0.6)
    factors = rng.normal(0, 0.01, size=(days, n_factors))
    specific = rng.normal(0, 0.008, size=(days, n))
    drift = rng.uniform(0.0002, 0.0008, size=n)
    rets = factors @ loadings.T + specific + drift
    return pd.DataFrame(rets, columns=names)

returns = load_returns(TICKERS)
tickers = list(returns.columns)
print(f"Universe: {len(tickers)} stocks, {len(returns)} days of returns")
returns.head()

# %% [markdown]
# ## 3. Expected returns and covariance
#
# * `mu` is the expected return of each stock.
# * `Sigma` is the covariance matrix: how much stocks move together.

# %%
mu = returns.mean().values * 252            # annualized expected return
Sigma = returns.cov().values * 252          # annualized covariance matrix
n = len(tickers)

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(np.corrcoef(returns.values.T), cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(n)); ax.set_xticklabels(tickers, rotation=90)
ax.set_yticks(range(n)); ax.set_yticklabels(tickers)
ax.set_title("Correlation between stocks")
fig.colorbar(im)
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 4. Formulate the QUBO
#
# Each stock `i` gets a binary variable `x_i` (1 = include, 0 = exclude).
#
# Objective (minimize):
#
# $$
# E(x) = q \cdot x^T \Sigma x \;-\; \mu^T x \;+\; A\,(\textstyle\sum_i x_i - K)^2
# $$
#
# * `q` is the risk-aversion weight (higher = more cautious).
# * `x^T Sigma x` is the risk of the chosen set (correlated holdings add risk).
# * `-mu^T x` rewards expected return.
# * `A (sum x_i - K)^2` is the **penalty** that pushes the portfolio toward exactly `K` stocks.
#
# Expanding the penalty gives a standard QUBO matrix `Q` and vector `c` such that
# `E(x) = x^T Q x + c^T x + const`.

# %%
K = 5          # number of stocks to hold
q = 0.5        # risk aversion
A = 2.0        # penalty strength (must be large enough to enforce the constraint)

def build_qubo(mu, Sigma, K, q, A):
    """Build the QUBO matrix Q (n x n) for minimizing x^T Q x."""
    n = len(mu)
    Q = q * Sigma.copy()
    # Linear return term and cardinality penalty A*(sum x - K)^2 on the diagonal and off-diagonal:
    # (sum x)^2 = sum_i x_i + 2 * sum_{i<j} x_i x_j  (since x_i^2 = x_i for binaries)
    # -2AK * sum x_i  +  A * (sum x)^2
    Q = Q + A * np.ones((n, n))
    np.fill_diagonal(Q, np.diag(Q))  # keep diagonal from (sum x)^2 term
    linear = -mu - 2 * A * K * np.ones(n)
    # Put linear terms on the diagonal (x_i^2 = x_i for binaries)
    Q = Q + np.diag(linear)
    return Q

Q = build_qubo(mu, Sigma, K, q, A)

def qubo_energy(x, Q):
    x = np.asarray(x, dtype=float)
    return float(x @ Q @ x + A * K * K)  # constant term A*K^2 added back for readability

# %% [markdown]
# ## 5. Classical benchmark: simulated annealing
#
# Simulated annealing starts from a random selection and makes small random changes,
# accepting worse moves early on (to escape local traps) and becoming stricter over time.

# %%
def simulated_annealing(Q, steps=20000, T0=1.0, T1=1e-3, seed=0):
    r = np.random.default_rng(seed)
    n = Q.shape[0]
    x = r.integers(0, 2, size=n)
    e = x @ Q @ x
    best_x, best_e = x.copy(), e
    for step in range(steps):
        T = T0 * (T1 / T0) ** (step / steps)   # geometric cooling schedule
        i = r.integers(n)
        x[i] ^= 1                              # flip one stock in/out
        e_new = x @ Q @ x
        if e_new < e or r.random() < np.exp((e - e_new) / T):
            e = e_new
            if e < best_e:
                best_e, best_x = e, x.copy()
        else:
            x[i] ^= 1                          # undo the move
    return best_x, best_e

t0 = time.time()
sa_best = None
for s in range(10):                            # several restarts for robustness
    x_s, e_s = simulated_annealing(Q, seed=s)
    if sa_best is None or e_s < sa_best[1]:
        sa_best = (x_s, e_s)
sa_time = time.time() - t0
sa_x, sa_e = sa_best
print("Simulated annealing selection:", [t for t, b in zip(tickers, sa_x) if b])
print(f"Selected: {sa_x.sum()} stocks, energy {sa_e:.4f}, time {sa_time:.2f}s")

# %% [markdown]
# ## 6. Ground truth by brute force (small problems only)
#
# With 20 stocks there are about one million combinations, which is small enough to check
# exhaustively. This gives us the true optimum to judge both heuristics against.

# %%
def brute_force(Q, n, K):
    best_x, best_e = None, np.inf
    for combo in itertools.combinations(range(n), K):   # only portfolios with exactly K stocks
        x = np.zeros(n, dtype=int)
        x[list(combo)] = 1
        e = x @ Q @ x
        if e < best_e:
            best_e, best_x = e, x
    return best_x, best_e

t0 = time.time()
bf_x, bf_e = brute_force(Q, n, K)
bf_time = time.time() - t0
print("Exact optimum:", [t for t, b in zip(tickers, bf_x) if b])
print(f"Energy {bf_e:.4f}, time {bf_time:.2f}s")
print("Simulated annealing gap to optimum:", round(sa_e - bf_e, 6))

# %% [markdown]
# ## 7. QAOA (quantum approach)
#
# QAOA alternates between two steps: a "cost" step that encodes the QUBO into quantum phases,
# and a "mixer" step that explores other combinations. A classical optimizer (COBYLA) tunes
# the step settings to find the lowest-energy outcome.
#
# QAOA on a simulator is limited to small problems (roughly 10 to 20 qubits). We therefore
# run QAOA on a smaller 8-stock subset, which is a realistic size for simulation today.

# %%
def run_qaoa_on_subset(subset_idx, K_sub, reps=2):
    """Run QAOA on a subset of stocks and return the selected tickers."""
    sub_mu = mu[subset_idx]
    sub_Sigma = Sigma[np.ix_(subset_idx, subset_idx)]
    sub_Q = build_qubo(sub_mu, sub_Sigma, K_sub, q, A)

    qp = QuadraticProgram("portfolio")
    for i in range(len(subset_idx)):
        qp.binary_var(name=f"x{i}")
    qp.minimize(quadratic=sub_Q)

    qubo = QuadraticProgramToQubo().convert(qp)
    qaoa = QAOA(sampler=Sampler(), optimizer=COBYLA(maxiter=200), reps=reps)
    result = MinimumEigenOptimizer(qaoa).solve(qubo)
    chosen = [tickers[subset_idx[i]] for i, v in enumerate(result.x) if v > 0.5]
    return chosen, result.fval

if HAVE_QISKIT:
    subset = list(range(8))
    t0 = time.time()
    qaoa_chosen, qaoa_val = run_qaoa_on_subset(subset, K_sub=3)
    print("QAOA selection (8-stock subset):", qaoa_chosen)
    print(f"Objective {qaoa_val:.4f}, time {time.time() - t0:.1f}s")
else:
    print("Qiskit not installed; skipping QAOA. Install with: pip install qiskit qiskit-algorithms qiskit-optimization")

# %% [markdown]
# ## 8. Scaling up: cluster, solve each cluster, then combine
#
# For hundreds or thousands of stocks, the QUBO becomes too large for current quantum hardware.
# The approach from the proposal:
#
# 1. Group stocks into clusters by correlation (similar stocks together).
# 2. Solve each cluster separately, choosing a small number of stocks from each.
# 3. Use a classical solver to pick the final portfolio from the cluster winners.

# %%
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

def cluster_stocks(returns_df, n_clusters):
    corr = np.corrcoef(returns_df.values.T)
    dist = np.sqrt(0.5 * (1 - corr))           # correlation distance
    np.fill_diagonal(dist, 0)
    Z = linkage(squareform(dist, checks=False), method="average")
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")
    return labels

def scaled_portfolio(returns_df, n_clusters=5, picks_per_cluster=1, K_final=5):
    labels = cluster_stocks(returns_df, n_clusters)
    mu_all = returns_df.mean().values * 252
    Sigma_all = returns_df.cov().values * 252
    winners = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        k_local = min(picks_per_cluster, len(idx))
        Qc = build_qubo(mu_all[idx], Sigma_all[np.ix_(idx, idx)], k_local, q, A)
        xc, _ = simulated_annealing(Qc, steps=5000, seed=int(c))  # stand-in for a quantum solver per cluster
        chosen_local = idx[xc.astype(bool)]
        winners.extend(chosen_local.tolist())
    # Final classical step: choose K_final among the winners
    winners = np.array(winners)
    Qf = build_qubo(mu_all[winners], Sigma_all[np.ix_(winners, winners)], K_final, q, A)
    xf, ef = simulated_annealing(Qf, steps=10000, seed=99)
    final_idx = winners[xf.astype(bool)]
    return [returns_df.columns[i] for i in final_idx], ef

final_portfolio, final_e = scaled_portfolio(returns, n_clusters=5, picks_per_cluster=2, K_final=5)
print("Scaled portfolio:", final_portfolio)
print(f"Final energy: {final_e:.4f}")

# %% [markdown]
# ## 9. Quick summary

# %%
summary = pd.DataFrame({
    "Method": ["Brute force (exact)", "Simulated annealing", "Cluster + SA (scaled)"],
    "Selected stocks": [
        ", ".join(t for t, b in zip(tickers, bf_x) if b),
        ", ".join(t for t, b in zip(tickers, sa_x) if b),
        ", ".join(final_portfolio),
    ],
    "Energy": [bf_e, sa_e, final_e],
})
summary

# %% [markdown]
# ## Notes
#
# * **Brute force** is exact but grows exponentially, so it is only feasible for about 20 to 25 stocks.
# * **Simulated annealing** is the classical benchmark. Compare its gap to the exact optimum.
# * **QAOA** shown here runs on a simulator. On real hardware, expect noise to reduce quality.
# * **Scaling** uses simulated annealing as a stand-in for a quantum solver per cluster. Swap in
#   QAOA or a quantum annealer (for example, D-Wave's `dwave-ocean-sdk`) when the cluster size fits.
# * Energies from different methods are only directly comparable when they use the same
#   `Q`, `K`, `q`, and `A`. The cluster method uses a different `K` per stage, so compare
#   its final selection by re-evaluating it on the full problem if needed.
