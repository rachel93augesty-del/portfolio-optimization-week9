# -----------------------------
# 1️⃣ Imports
# -----------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cvxpy as cp  # CVXPY for optimization

# -----------------------------
# 2️⃣ Load Data
# -----------------------------
def load_data(all_assets_path):
    """Load all assets prices/forecasted prices from single CSV"""
    df = pd.read_csv(all_assets_path, parse_dates=['Date'], index_col='Date')
    return df

# -----------------------------
# 3️⃣ Compute Expected Returns
# -----------------------------
def compute_expected_returns(df):
    """Compute annualized expected returns from price data"""
    daily_returns = df.pct_change().dropna()
    exp_returns = daily_returns.mean() * 252  # annualized
    return exp_returns

# -----------------------------
# 4️⃣ Compute Covariance Matrix
# -----------------------------
def compute_cov_matrix(df):
    """Compute annualized covariance matrix of asset returns"""
    daily_returns = df.pct_change().dropna()
    cov_matrix = daily_returns.cov() * 252  # annualized
    return cov_matrix

# -----------------------------
# 5️⃣ Generate Efficient Frontier
# -----------------------------
def generate_efficient_frontier(mu: pd.Series, Sigma: pd.DataFrame, points: int = 50):
    """
    Generate Efficient Frontier using CVXPY
    :param mu: Annualized expected returns vector
    :param Sigma: Annualized covariance matrix
    :param points: Number of portfolios to simulate
    :return: risks, returns, weights
    """
    n = len(mu)
    w = cp.Variable(n)
    risks, returns, weights = [], [], []

    target_returns_range = np.linspace(mu.min(), mu.max(), points)

    for tr in target_returns_range:
        constraints = [cp.sum(w) == 1, w >= 0, mu.values @ w == tr]
        risk = cp.quad_form(w, Sigma.values)
        prob = cp.Problem(cp.Minimize(risk), constraints)
        prob.solve(solver=cp.SCS, verbose=False)

        if w.value is not None:
            risks.append(np.sqrt(prob.value))
            returns.append(tr)
            weights.append(w.value)
        else:
            risks.append(np.nan)
            returns.append(tr)
            weights.append([np.nan]*n)

    return np.array(risks), np.array(returns), np.array(weights)

# -----------------------------
# 6️⃣ Plotting Functions
# -----------------------------
def plot_efficient_frontier(risks: np.ndarray, returns: np.ndarray, title: str = "Efficient Frontier"):
    plt.figure(figsize=(10,6))
    plt.plot(risks, returns, 'b-', linewidth=2)
    plt.xlabel("Portfolio Risk (Std Dev)")
    plt.ylabel("Portfolio Return")
    plt.title(title)
    plt.grid(True)
    plt.show()

def plot_cov_matrix(cov_matrix):
    plt.figure(figsize=(6,5))
    sns.heatmap(cov_matrix, annot=True, cmap="coolwarm", fmt=".4f")
    plt.title("Covariance Matrix Heatmap")
    plt.show()

# -----------------------------
# 7️⃣ Main Execution
# -----------------------------
if __name__ == "__main__":
    all_assets_path = "data/processed/all_assets_clean.csv"

    # Load data
    all_assets_df = load_data(all_assets_path)

    # Compute expected returns and covariance
    exp_returns = compute_expected_returns(all_assets_df)
    cov_matrix = compute_cov_matrix(all_assets_df)

    print("Expected Returns (Annualized):\n", exp_returns)
    print("\nCovariance Matrix:\n", cov_matrix)

    # Generate Efficient Frontier
    risks, returns, weights = generate_efficient_frontier(exp_returns, cov_matrix, points=50)

    # Plot results
    plot_efficient_frontier(risks, returns)
    plot_cov_matrix(cov_matrix)
