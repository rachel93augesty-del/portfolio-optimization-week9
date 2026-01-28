import matplotlib.pyplot as plt
import seaborn as sns

def plot_portfolio_cumulative(cumulative_returns, title="Portfolio Cumulative Returns"):
    plt.figure(figsize=(10,6))
    plt.plot(cumulative_returns, color='blue', lw=2)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Cumulative Returns")
    plt.grid(True)
    plt.show()

def plot_covariance_heatmap(cov_matrix, title="Covariance Matrix Heatmap"):
    plt.figure(figsize=(8,6))
    sns.heatmap(cov_matrix, annot=True, fmt=".4f", cmap="coolwarm")
    plt.title(title)
    plt.show()
